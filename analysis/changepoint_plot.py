"""Headline chart: the process change in yield and in sensor behaviour."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from snow import query

CP = pd.Timestamp("2008-08-20 17:36:00")
WINDOW = 100
Path("out/charts").mkdir(parents=True, exist_ok=True)

df = query("""
    SELECT wafer_id, measured_at, is_fail, sensor_060
    FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES ORDER BY measured_at
""")
df.columns = [c.lower() for c in df.columns]

pre, post = df[df.measured_at < CP], df[df.measured_at >= CP]
fr_pre, fr_post = pre.is_fail.mean(), post.is_fail.mean()

fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                       gridspec_kw={"height_ratios": [1, 1], "hspace": 0.25})

# --- yield ---------------------------------------------------------------
roll = df.is_fail.rolling(WINDOW, center=True).mean()
ax[0].plot(df.measured_at, roll, color="#333", lw=1.3,
           label=f"{WINDOW}-wafer rolling failure rate")
ax[0].hlines(fr_pre, df.measured_at.min(), CP,
             color="steelblue", lw=2.5, label=f"pre: {fr_pre:.1%}")
ax[0].hlines(fr_post, CP, df.measured_at.max(),
             color="seagreen", lw=2.5, label=f"post: {fr_post:.1%}")
ax[0].axvline(CP, color="crimson", ls="--", lw=1.6)
ax[0].set_ylabel("failure rate")
ax[0].set_title(f"Process change {CP:%d %B %Y} — failure rate "
                f"{fr_pre:.1%} → {fr_post:.1%}", fontsize=13)
ax[0].legend(fontsize=9, loc="upper right")
ax[0].set_ylim(bottom=0)

# --- the signal the changepoint was actually found from ------------------
ax[1].plot(df.measured_at, df.sensor_060, lw=0.6, color="#333")
ax[1].axvline(CP, color="crimson", ls="--", lw=1.6)
ax[1].set_ylabel("SENSOR_060")
ax[1].set_xlabel("")
ax[1].set_title("Changepoint located from sensor behaviour alone — "
                "labels not used", fontsize=10)

# annotate the shift in dispersion
for seg, lbl, xoff in [(pre, "pre", 0.18), (post, "post", 0.65)]:
    ax[1].text(xoff, 0.88, f"σ = {seg.sensor_060.std():.1f}",
               transform=ax[1].transAxes, fontsize=9, color="steelblue",
               bbox=dict(fc="white", ec="steelblue", alpha=0.8, pad=3))

fig.tight_layout()
fig.savefig("out/charts/changepoint.png", dpi=140, bbox_inches="tight")
print(f"pre  {len(pre):4d} wafers, {fr_pre:.2%} fail, σ={pre.sensor_060.std():.2f}")
print(f"post {len(post):4d} wafers, {fr_post:.2%} fail, σ={post.sensor_060.std():.2f}")
print("wrote out/charts/changepoint.png")