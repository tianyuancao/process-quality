import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from build_dataset import X_train, y_train

rows = []
for c in X_train.columns:
    a = X_train.loc[y_train == 1, c].dropna()
    b = X_train.loc[y_train == 0, c].dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial effect size
    eff = 2 * u / (len(a) * len(b)) - 1
    rows.append({"feature": c, "p": p, "effect": eff,
                 "median_fail": a.median(), "median_pass": b.median()})

df = pd.DataFrame(rows).sort_values("p")
df["p_fdr"] = multipletests(df["p"], method="fdr_bh")[1]

print(f"tested {len(df)} sensors")
print(f"p < 0.05 raw:        {(df.p < 0.05).sum()}  "
      f"(expected by chance: {0.05*len(df):.0f})")
print(f"survives FDR q<0.10: {(df.p_fdr < 0.10).sum()}")
print(df.head(10).round(4).to_string(index=False))

df.to_csv("out/univariate_screening.csv", index=False)