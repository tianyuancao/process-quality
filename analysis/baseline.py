import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import average_precision_score

from build_dataset import X_train, X_test, y_train, y_test

print(f"no-skill PR-AUC (train prevalence) = {y_train.mean():.4f}")

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, class_weight="balanced")),
])

cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                         scoring="average_precision")
print(f"logreg CV PR-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

pipe.fit(X_train, y_train)
test_ap = average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])
print(f"logreg test PR-AUC = {test_ap:.4f}")