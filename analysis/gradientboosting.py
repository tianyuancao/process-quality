"""Gradient boosting on SECOM, benchmarked against the logistic baseline.

Chronological CV throughout. PR-AUC only — accuracy is meaningless at 6.9%
prevalence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_dataset import X_train, X_test, y_train, y_test

import pandas as pd
sig = pd.read_csv("out/univariate_screening.csv")
keep = sig.loc[sig.p_fdr < 0.10, "feature"].tolist()

RANDOM_STATE = 42
CV = TimeSeriesSplit(n_splits=5)
NO_SKILL = y_train.mean()

Path("out").mkdir(exist_ok=True)
results = {}

X_train = X_train[keep]
X_test  = X_test[keep]
# y_train, y_test stay exactly as they are

def evaluate(name, model):
    """Cross-validate on train, then score once on the held-out test set."""
    scores = cross_val_score(model, X_train, y_train, cv=CV,
                             scoring="average_precision", n_jobs=1)
    model.fit(X_train, y_train)
    test_ap = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    results[name] = {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "cv_folds": [float(s) for s in scores],
        "test_ap": float(test_ap),
        "lift_vs_no_skill": float(scores.mean() / NO_SKILL),
    }
    print(f"{name:22s} CV {scores.mean():.4f} ± {scores.std():.4f}   "
          f"test {test_ap:.4f}")
    return model


print(f"train {len(X_train)} rows, {y_train.mean():.2%} fail")
print(f"test  {len(X_test)} rows, {y_test.mean():.2%} fail")
print(f"no-skill PR-AUC = {NO_SKILL:.4f}\n")

# --- baseline ------------------------------------------------------------
logreg = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
evaluate("logistic regression", logreg)


# --- gradient boosting ---------------------------------------------------
# HistGradientBoosting is scikit-learn's LightGBM-style implementation.
# No imputer or scaler: it learns a default direction for missing values
# and trees are scale-invariant. No OpenMP dependency, unlike LightGBM.
gbm = HistGradientBoostingClassifier(
    max_iter=400,
    max_depth=3,
    max_leaf_nodes=7,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
gbm = evaluate("hist gradient boosting", gbm)

# --- sanity check: shuffled target should score at no-skill --------------
rng = np.random.default_rng(RANDOM_STATE)
y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
shuffled = cross_val_score(
    HistGradientBoostingClassifier(**gbm.get_params()),
    X_train, y_shuffled, cv=CV, scoring="average_precision",
)
print(f"\nshuffled-target CV  {shuffled.mean():.4f} "
      f"(should sit near {NO_SKILL:.4f})")
results["shuffled_control"] = {"cv_mean": float(shuffled.mean())}

# --- verdict -------------------------------------------------------------
base, gb = results["logistic regression"], results["hist gradient boosting"]
gain = gb["cv_mean"] - base["cv_mean"]
pooled_std = max(gb["cv_std"], base["cv_std"])

print("\n--- verdict ---")
print(f"CV gain over baseline: {gain:+.4f}  (fold std {pooled_std:.4f})")
if gain < pooled_std:
    print("Gain is within fold noise. Treat as NO real improvement.")
else:
    print("Gain exceeds fold-to-fold variation. Modest but real.")

Path("out/model_results.json").write_text(json.dumps(results, indent=2))
print("wrote out/model_results.json")


# --- threshold behaviour on the test set ---------------------------------
proba = gbm.predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, proba)
print(f"\ntest set holds {int(y_test.sum())} failures — "
      "every metric below is coarse")
for target in (0.20, 0.30, 0.50):
    ok = prec[:-1] >= target
    if ok.any():
        i = int(np.argmax(rec[:-1] * ok))
        print(f"  precision ≥{target:.0%}: recall {rec[i]:.2f} "
              f"at threshold {thr[i]:.3f}")
    else:
        print(f"  precision ≥{target:.0%}: not reachable")

# --- persist for Week 3 --------------------------------------------------
from sklearn.inspection import permutation_importance

pi = permutation_importance(gbm, X_test, y_test, n_repeats=10,
                            scoring="average_precision",
                            random_state=RANDOM_STATE, n_jobs=1)
pd.DataFrame({
    "feature": X_train.columns,
    "importance": pi.importances_mean,
    "std": pi.importances_std,
}).sort_values("importance", ascending=False).to_csv(
    "out/permutation_importance.csv", index=False)