"""Profile the raw SECOM sensors to decide staging rules.

Outputs a per-column summary and the column lists consumed by dbt.
"""

import json
from pathlib import Path
import pandas as pd
from snow import query

NULL_DROP_THRESHOLD = 0.50    # drop columns with more nulls than this
NEAR_CONST_THRESHOLD = 0.99   # drop if one value covers this share of non-nulls

df = query("SELECT * FROM SECOM.RAW.SECOM_RAW ORDER BY WAFER_ID")
sensors = [c for c in df.columns if c.startswith("SENSOR_")]
X = df[sensors]

print(f"rows={len(df)}  sensors={len(sensors)}  fail_rate={df.IS_FAIL.mean():.3%}")

# ---- per-column profile -------------------------------------------------
prof = pd.DataFrame({
    "null_frac":  X.isna().mean(),
    "n_unique":   X.nunique(dropna=True),
    "std":        X.std(),
    "mean":       X.mean(),
    "min":        X.min(),
    "max":        X.max(),
})

# share of non-null rows taken by the single most common value
prof["top_value_frac"] = X.apply(
    lambda s: s.value_counts(dropna=True).iloc[0] / s.notna().sum()
    if s.notna().any() else 1.0
)

prof["all_null"]    = prof.null_frac == 1.0
prof["constant"]    = prof.n_unique <= 1
prof["near_const"]  = (~prof.constant) & (prof.top_value_frac >= NEAR_CONST_THRESHOLD)
prof["high_null"]   = prof.null_frac > NULL_DROP_THRESHOLD
prof["drop"]        = prof.all_null | prof.constant | prof.near_const | prof.high_null
# pairwise redundancy — not detectable column by column
drop_redundant = set(pd.read_csv("out/drop_redundant.csv").column_name)
prof["redundant"] = prof.index.str.lower().isin(drop_redundant)
prof["drop"] = prof["drop"] | prof["redundant"]
# ---- summary ------------------------------------------------------------
print("\n--- drop reasons (overlapping) ---")
for flag in ["all_null", "constant", "near_const", "high_null"]:
    print(f"{flag:12s} {prof[flag].sum():4d}")
print(f"{'DROP total':12s} {prof['drop'].sum():4d}")
print(f"{'KEEP':12s} {(~prof['drop']).sum():4d}")

print("\n--- null distribution across all sensors ---")
print(prof.null_frac.describe(percentiles=[.5, .75, .9, .95, .99]).round(3))

print("\n--- widest scale range among kept columns ---")
kept = prof[~prof["drop"]]
print(kept["std"].describe().round(3))

# ---- outputs ------------------------------------------------------------
Path("out").mkdir(exist_ok=True)
prof.round(6).to_csv("out/sensor_profile.csv")

keep_cols = kept.index.tolist()
Path("out/keep_columns.json").write_text(json.dumps(keep_cols, indent=1))
print(f"\nwrote out/sensor_profile.csv and out/keep_columns.json ({len(keep_cols)} kept)")

# ---- save outputs to dbt seeds ---------------------
cols = json.load(open("out/keep_columns.json"))
pd.DataFrame({"column_name": cols}).to_csv(
    "../dbt/secom/seeds/kept_sensors.csv", index=False
)