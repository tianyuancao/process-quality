"""I-MR control charts on SECOM, corrected for the late-August regime shift.

The first pass computed Phase I limits on a window that was not in control,
and evaluated lift across the whole series. Because the regime change
coincides with a drop in failure rate, "violating" was largely a proxy for
"late period" and lift came out inverted. This version detects the
changepoint, charts each regime against its own in-control reference, and
computes lift within regime so calendar time is controlled for.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from snow import query

D2, D4 = 1.128, 3.267        # moving-range constants, n = 2
PHASE1_FRAC = 0.30           # share of each regime used as reference
ANCHOR = "sensor_060"        # strongest univariate signal; defines the split

Path("out/charts").mkdir(parents=True, exist_ok=True)

SENSORS = ["sensor_060", "sensor_248", "sensor_520", "sensor_104", "sensor_511"]

df = query(f"""
    SELECT wafer_id, measured_at, is_fail, {", ".join(SENSORS)}
    FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES
    ORDER BY measured_at
""")
df.columns = [c.lower() for c in df.columns]
df = df.reset_index(drop=True)


# ---------------------------------------------------------------- changepoint
def find_changepoint(x):
    """Single changepoint by exhaustive search on the two-segment SSE.

    Equivalent to ruptures' Binseg with n_bkps=1 for an l2 cost, but with no
    extra dependency. Operates on standardised values so scale is irrelevant.
    """
    s = pd.Series(x).ffill().bfill()
    s = (s - s.mean()) / s.std()
    v = s.values
    n = len(v)
    lo, hi = int(0.15 * n), int(0.85 * n)      # ignore edges

    csum = np.concatenate([[0], np.cumsum(v)])
    csq = np.concatenate([[0], np.cumsum(v ** 2)])

    def sse(a, b):
        k = b - a
        total = csq[b] - csq[a]
        return total - (csum[b] - csum[a]) ** 2 / k

    costs = [sse(0, i) + sse(i, n) for i in range(lo, hi)]
    return lo + int(np.argmin(costs))


CP = find_changepoint(df[ANCHOR])
print(f"changepoint at row {CP} — {df.measured_at.iloc[CP]}")
print(f"  pre : {CP} wafers, fail rate {df.is_fail[:CP].mean():.2%}")
print(f"  post: {len(df)-CP} wafers, fail rate {df.is_fail[CP:].mean():.2%}\n")


# ------------------------------------------------------------------ I-MR core
def imr_limits(x):
    x = pd.Series(x).dropna()
    mr = x.diff().abs().dropna()
    mrbar, xbar = mr.mean(), x.mean()
    sigma = mrbar / D2
    return {"center": xbar, "sigma": sigma,
            "ucl": xbar + 3 * sigma, "lcl": xbar - 3 * sigma,
            "mr_center": mrbar, "mr_ucl": D4 * mrbar}


def nelson(x, lim):
    """Nelson rules 1-4. Rule 2 flags only the 9th point of a run onward."""
    c, s = lim["center"], lim["sigma"]
    above = x > c
    out = pd.DataFrame(index=x.index)

    out["r1_beyond_3s"] = (x - c).abs() > 3 * s

    run = above.ne(above.shift()).cumsum()
    size = above.groupby(run).transform("size")
    pos = above.groupby(run).cumcount() + 1
    out["r2_run_of_9"] = size.ge(9) & pos.ge(9) & x.notna()

    d = np.sign(x.diff())
    grp = d.ne(d.shift()).cumsum()
    tsize = d.groupby(grp).transform("size")
    tpos = d.groupby(grp).cumcount() + 1
    out["r3_trend_of_6"] = tsize.ge(6) & tpos.ge(6) & d.ne(0)

    beyond2 = (x - c).abs() > 2 * s
    hi = (beyond2 & above).rolling(3).sum().ge(2)
    lo = (beyond2 & ~above).rolling(3).sum().ge(2)
    out["r4_2of3_beyond_2s"] = (hi | lo).fillna(False)

    return out.fillna(False)


def phase1_iterate(x, frac=PHASE1_FRAC, max_rounds=5):
    """Establish limits, drop rule-1 outliers, recompute. Montgomery's
    iterative Phase I: the reference window must itself be in control."""
    ref = pd.Series(x).iloc[:max(int(len(x) * frac), 25)].dropna()
    for _ in range(max_rounds):
        lim = imr_limits(ref)
        keep = ref.between(lim["lcl"], lim["ucl"])
        if keep.all() or keep.sum() < 20:
            break
        ref = ref[keep]
    return imr_limits(ref), len(ref)


# -------------------------------------------------------------------- per-run
regimes = [("pre", df.iloc[:CP]), ("post", df.iloc[CP:])]
summary = []

for col in SENSORS:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    all_flags = pd.Series(False, index=df.index)

    for label, seg in regimes:
        x = seg[col]
        lim, n_ref = phase1_iterate(x)
        flags = nelson(x, lim)
        viol = flags.any(axis=1)
        all_flags.loc[seg.index] = viol

        # lift computed WITHIN this regime only
        fr_v = seg.loc[viol, "is_fail"].mean()
        fr_c = seg.loc[~viol, "is_fail"].mean()
        summary.append({
            "sensor": col, "regime": label,
            "n": len(seg), "n_violations": int(viol.sum()),
            "pct_violating": viol.mean(),
            "fail_rate_violating": fr_v,
            "fail_rate_clean": fr_c,
            "lift": fr_v / fr_c if fr_c else np.nan,
            "phase1_n": n_ref,
            **{r: int(flags[r].sum()) for r in flags.columns},
        })

        t = seg.measured_at
        axes[0].plot(t, x, lw=0.7, color="#444")
        axes[0].scatter(t[viol], x[viol], s=16, color="crimson", zorder=3)
        for key, ls in [("center", "-"), ("ucl", "--"), ("lcl", "--")]:
            axes[0].plot(t, [lim[key]] * len(t), ls=ls, lw=1.0,
                         color="steelblue")
        mr = x.diff().abs()
        axes[1].plot(t, mr, lw=0.7, color="#444")
        axes[1].plot(t, [lim["mr_center"]] * len(t), lw=1.0, color="steelblue")
        axes[1].plot(t, [lim["mr_ucl"]] * len(t), ls="--", lw=1.0,
                     color="steelblue")

    cp_time = df.measured_at.iloc[CP]
    for ax in axes:
        ax.axvline(cp_time, color="green", ls=":", lw=1.5)

    fails = df[df.is_fail == 1]
    axes[0].scatter(fails.measured_at, [axes[0].get_ylim()[0]] * len(fails),
                    marker="|", color="orange", s=40, label="failed wafer")
    axes[0].set_title(f"{col.upper()} — I-MR, limits per regime "
                      f"(changepoint {cp_time:%Y-%m-%d})")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Moving range")

    fig.tight_layout()
    fig.savefig(f"out/charts/imr_2_{col}.png", dpi=130)
    plt.close(fig)


res = pd.DataFrame(summary)
print(res[["sensor", "regime", "n", "n_violations", "pct_violating",
           "fail_rate_violating", "fail_rate_clean", "lift"]]
      .round(4).to_string(index=False))
res.round(4).to_csv("out/control_chart_summary.csv", index=False)

print("\nrule breakdown")
print(res.groupby("regime")[["r1_beyond_3s", "r2_run_of_9",
                             "r3_trend_of_6", "r4_2of3_beyond_2s"]]
      .sum().to_string())
print("\ncharts written to out/charts/")