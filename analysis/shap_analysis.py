"""SHAP attribution within the post-change regime.

Scoped to the post-regime process only: the changepoint analysis showed
the two periods behave as different processes, so explaining a model
fitted across the boundary would contradict that finding.

The purpose here is not a strong predictor — the model is weak and
reported as such. It is convergence: do SHAP rankings agree with the
univariate screening and the control charts?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from snow import query

CHANGEPOINT = "2008-08-20 17:36:00"
RANDOM_STATE = 42
Path("out/charts").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ features
screen = pd.read_csv("out/univariate_screening.csv")
feats = screen.loc[screen.p_fdr < 0.10, "feature"].str.lower().tolist()
print(f"{len(feats)} FDR-surviving sensors")

df = query(f"""
    SELECT wafer_id, measured_at, is_fail, {", ".join(feats)}
    FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES
    ORDER BY measured_at
""")
df.columns = [c.lower() for c in df.columns]

post = df[df.measured_at >= CHANGEPOINT].reset_index(drop=True)
print(f"post-regime: {len(post)} wafers, {int(post.is_fail.sum())} failures "
      f"({post.is_fail.mean():.2%})")

X = post[feats]
y = post["is_fail"].astype(int)
cut = int(len(post) * 0.8)
X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
print(f"train {len(X_tr)} ({int(y_tr.sum())} fails) | "
      f"test {len(X_te)} ({int(y_te.sum())} fails)")

# --------------------------------------------------------------------- model
model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=3, max_leaf_nodes=7,
    learning_rate=0.05, min_samples_leaf=15,
    l2_regularization=1.0, class_weight="balanced",
    random_state=RANDOM_STATE,
)

cv = cross_val_score(model, X_tr, y_tr, cv=TimeSeriesSplit(n_splits=4),
                     scoring="average_precision")
model.fit(X_tr, y_tr)
test_ap = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])

print(f"\nno-skill PR-AUC = {y_tr.mean():.4f}")
print(f"CV PR-AUC       = {cv.mean():.4f} ± {cv.std():.4f}")
print(f"test PR-AUC     = {test_ap:.4f}")
print("Model is weak — SHAP is used for attribution, not to claim accuracy.")

# ---------------------------------------------------------------------- SHAP
# Permutation explainer: model-agnostic, works with HistGradientBoosting's
# native NaN handling. Background sampled to keep runtime sane.
bg = shap.utils.sample(X_tr, 100, random_state=RANDOM_STATE)
explainer = shap.Explainer(
    lambda d: model.predict_proba(d)[:, 1], bg,
    feature_names=feats,
)
sv = explainer(X_te)

mean_abs = np.abs(sv.values).mean(axis=0)
rank = (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True))
rank["shap_rank"] = rank.index + 1
print("\n--- top 10 by mean |SHAP| ---")
print(rank.head(10).round(5).to_string(index=False))

# ------------------------------------------------------- convergence check
screen["feature"] = screen.feature.str.lower()
screen["univariate_rank"] = screen.p.rank().astype(int)
merged = rank.merge(screen[["feature", "univariate_rank", "p_fdr", "effect"]],
                    on="feature")

rho = merged.shap_rank.corr(merged.univariate_rank, method="spearman")
print(f"\nSpearman rank correlation, SHAP vs univariate: {rho:.3f}")

for s in ["sensor_060", "sensor_248", "sensor_520", "sensor_104", "sensor_511"]:
    if s in merged.feature.values:
        r = merged.loc[merged.feature == s].iloc[0]
        print(f"  {s}: SHAP rank {int(r.shap_rank):2d}, "
              f"univariate rank {int(r.univariate_rank):2d}")

merged.round(5).to_csv("out/shap_vs_univariate.csv", index=False)

# --------------------------------------------------------------------- plots
shap.plots.bar(sv, max_display=15, show=False)
plt.title("Mean |SHAP| — post-regime")
plt.tight_layout()
plt.savefig("out/charts/shap_bar.png", dpi=130)
plt.close()

shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP value distribution — post-regime")
plt.tight_layout()
plt.savefig("out/charts/shap_beeswarm.png", dpi=130)
plt.close()

# single failed wafer, highest predicted risk — the interview plot
proba = model.predict_proba(X_te)[:, 1]
fails = np.where(y_te.values == 1)[0]
if len(fails):
    i = int(fails[np.argmax(proba[fails])])
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"Wafer {post.wafer_id.iloc[cut + i]} — failed, "
              f"p={proba[i]:.3f}")
    plt.tight_layout()
    plt.savefig("out/charts/shap_waterfall_failed.png", dpi=130)
    plt.close()
    print(f"\nwaterfall: wafer {post.wafer_id.iloc[cut+i]}, p={proba[i]:.3f}")

json.dump({
    "post_regime_n": len(post),
    "post_regime_fails": int(post.is_fail.sum()),
    "cv_pr_auc": float(cv.mean()), "cv_std": float(cv.std()),
    "test_pr_auc": float(test_ap), "no_skill": float(y_tr.mean()),
    "spearman_shap_vs_univariate": float(rho),
}, open("out/shap_results.json", "w"), indent=2)

print("\nwrote out/charts/shap_*.png, out/shap_vs_univariate.csv")