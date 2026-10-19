"""Pairwise redundancy across all kept sensors.

Profiling checked single-column properties (constant, near-constant, null
fraction). Redundancy is a pairwise property and was missed: SENSOR_520 turned
out to be a linear transform of SENSOR_248. This finds every such pair.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from snow import query

RHO = 0.999          # rank-correlation threshold for "same measurement"
OUT = Path("out"); OUT.mkdir(exist_ok=True)

df = query("SELECT * FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES ORDER BY measured_at")
df.columns = [c.lower() for c in df.columns]
X = df[[c for c in df.columns if c.startswith("sensor_")]]
print(f"{X.shape[1]} sensors, {len(X)} wafers")

# Spearman on pairwise-complete data. Rank correlation catches monotonic
# transforms (log, unit conversion), which Pearson would miss.
corr = X.corr(method="spearman", min_periods=30).abs()
np.fill_diagonal(corr.values, np.nan)

pairs = (corr.stack()
         .reset_index()
         .rename(columns={"level_0": "a", "level_1": "b", 0: "rho"}))
pairs = pairs[pairs.a < pairs.b]                    # dedupe symmetric entries
dup = pairs[pairs.rho >= RHO].sort_values("rho", ascending=False)

print(f"\n{len(dup)} pairs with |Spearman| >= {RHO}")

rows = []
for _, r in dup.iterrows():
    m = X[[r.a, r.b]].dropna()
    if len(m) < 30:
        continue
    slope, intercept = np.polyfit(m[r.a], m[r.b], 1)
    resid = m[r.b] - (slope * m[r.a] + intercept)
    r2 = 1 - resid.var() / m[r.b].var()
    rows.append({
        "a": r.a, "b": r.b, "rho": round(r.rho, 6),
        "slope": slope, "intercept": intercept, "linear_r2": r2,
        "same_null_pattern": bool((X[r.a].isna() == X[r.b].isna()).all()),
        "n_common": len(m),
        "relation": ("identical" if np.allclose(m[r.a], m[r.b])
                     else "linear" if r2 > 0.9999
                     else "monotonic"),
    })

rel = pd.DataFrame(rows)
if len(rel):
    print(rel.round(4).to_string(index=False))
    print("\nrelation types:")
    print(rel.relation.value_counts().to_string())

# --- greedy dedup: keep one member of each connected group -----------------
import itertools
groups, seen = [], set()
adj = {}
for _, r in rel.iterrows():
    adj.setdefault(r.a, set()).add(r.b)
    adj.setdefault(r.b, set()).add(r.a)

for node in adj:
    if node in seen:
        continue
    stack, comp = [node], set()
    while stack:
        n = stack.pop()
        if n in comp:
            continue
        comp.add(n); seen.add(n)
        stack.extend(adj.get(n, []))
    groups.append(sorted(comp))

drop = []
for g in groups:
    drop.extend(g[1:])          # keep the first, drop the rest
    print(f"\ngroup: {g}  → keep {g[0]}")

print(f"\n{len(groups)} redundant groups, {len(drop)} columns to drop")
print(f"{X.shape[1]} → {X.shape[1] - len(drop)} sensors")

rel.round(6).to_csv(OUT / "redundant_pairs.csv", index=False)
pd.Series(drop, name="column_name").to_csv(OUT / "drop_redundant.csv",
                                           index=False)
print("\nwrote out/redundant_pairs.csv, out/drop_redundant.csv")