"""Statistical backing for the SENSOR_060 rule, plus a duplicate-column check."""

import numpy as np
import pandas as pd
from scipy import stats
from snow import query

SENSORS = ["sensor_060", "sensor_248", "sensor_520", "sensor_104", "sensor_511"]

df = query(f"""
    SELECT wafer_id, measured_at, is_fail, {", ".join(SENSORS)}
    FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES ORDER BY measured_at
""")
df.columns = [c.lower() for c in df.columns]

import numpy as np
m = df[["sensor_248", "sensor_520"]].dropna()
print(np.polyfit(m.sensor_248, m.sensor_520, 1))   # linear?
print(m.sensor_520.div(m.sensor_248).describe())    # constant ratio?
print((df.sensor_248.isna() == df.sensor_520.isna()).all())  # same nulls?



# --- 1. are 248 and 520 the same column? --------------------------------
print("--- correlation among flagged sensors ---")
corr = df[SENSORS].corr(method="spearman").round(3)
print(corr.to_string())

pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(SENSORS)
         for b in SENSORS[i + 1:] if abs(corr.loc[a, b]) > 0.95]
for a, b, r in pairs:
    ident = np.allclose(df[a].fillna(-999), df[b].fillna(-999))
    print(f"\n{a} vs {b}: rho={r:.4f}, identical={ident}")

# --- 2. Fisher exact on the SENSOR_060 post-regime rule -----------------
# 83 flagged / 10 failed, 1151 clean / 46 failed
table = [[10, 73], [46, 1105]]
odds, p = stats.fisher_exact(table, alternative="greater")
print(f"\n--- SENSOR_060 rule, post regime ---")
print(f"odds ratio = {odds:.2f}   one-sided p = {p:.4f}")

# Wilson CI on the flagged failure rate
k, n = 10, 83
lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95,
                                             method="wilson")
print(f"flagged fail rate = {k/n:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
print(f"baseline          = {46/1151:.1%}")