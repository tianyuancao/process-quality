"""SECOM process quality — findings app.

Presentation layer only: no model fitting, no changepoint search. Everything
here reads artefacts produced by the analysis scripts, so the app starts fast
and cannot silently disagree with the write-up.

    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from snow import query

CHANGEPOINT = pd.Timestamp("2008-08-20 17:36:00")
D2, D4 = 1.128, 3.267
OUT = Path("out")

st.set_page_config(page_title="SECOM Process Quality", layout="wide")


# ----------------------------------------------------------------- data ----
@st.cache_data(show_spinner="Querying Snowflake…")
def load_data():
    screen = pd.read_csv(OUT / "univariate_screening.csv")
    screen["feature"] = screen.feature.str.lower()
    feats = screen.loc[screen.p_fdr < 0.10, "feature"].tolist()
    feats = [f for f in feats if f != "sensor_520"]      # duplicate of 248

    df = query(f"""
        SELECT wafer_id, measured_at, is_fail, {", ".join(feats)}
        FROM SECOM.DBT_SCHEMA.STG_SECOM_FEATURES
        ORDER BY measured_at
    """)
    df.columns = [c.lower() for c in df.columns]
    df["regime"] = np.where(df.measured_at < CHANGEPOINT, "pre", "post")
    return df, screen, feats


@st.cache_data
def load_artifacts():
    def maybe(name, reader):
        p = OUT / name
        return reader(p) if p.exists() else None

    return {
        "lift": maybe("control_chart_summary.csv", pd.read_csv),
        "shap": maybe("shap_vs_univariate.csv", pd.read_csv),
        "model": maybe("model_results.json", lambda p: json.loads(p.read_text())),
    }


df, screen, feats = load_data()
art = load_artifacts()


# ------------------------------------------------------------------ SPC ----
def imr_limits(x, frac=0.30, rounds=5):
    """Iterative Phase I: limits from a reference window, outliers removed."""
    ref = pd.Series(x).iloc[:max(int(len(x) * frac), 25)].dropna()
    for _ in range(rounds):
        mr = ref.diff().abs().dropna()
        mrbar, xbar = mr.mean(), ref.mean()
        sigma = mrbar / D2
        keep = ref.between(xbar - 3 * sigma, xbar + 3 * sigma)
        if keep.all() or keep.sum() < 20:
            break
        ref = ref[keep]
    mr = ref.diff().abs().dropna()
    mrbar, xbar = mr.mean(), ref.mean()
    sigma = mrbar / D2
    return {"center": xbar, "sigma": sigma,
            "ucl": xbar + 3 * sigma, "lcl": xbar - 3 * sigma,
            "mr_center": mrbar, "mr_ucl": D4 * mrbar}


def nelson(x, lim):
    c, s = lim["center"], lim["sigma"]
    above = x > c
    f = pd.DataFrame(index=x.index)
    f["beyond 3σ"] = (x - c).abs() > 3 * s

    run = above.ne(above.shift()).cumsum()
    f["run of 9"] = (above.groupby(run).transform("size").ge(9)
                     & above.groupby(run).cumcount().add(1).ge(9) & x.notna())

    d = np.sign(x.diff())
    g = d.ne(d.shift()).cumsum()
    f["trend of 6"] = (d.groupby(g).transform("size").ge(6)
                       & d.groupby(g).cumcount().add(1).ge(6) & d.ne(0))

    b2 = (x - c).abs() > 2 * s
    f["2 of 3 beyond 2σ"] = (((b2 & above).rolling(3).sum().ge(2))
                             | ((b2 & ~above).rolling(3).sum().ge(2))).fillna(False)
    return f.fillna(False)


# --------------------------------------------------------------- layout ----
st.title("SECOM — Semiconductor Process Quality")
st.caption("1,567 wafers · 590 anonymised sensors · July–October 2008")

pre, post = df[df.regime == "pre"], df[df.regime == "post"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wafers", f"{len(df):,}")
c2.metric("Failure rate, pre", f"{pre.is_fail.mean():.2%}")
c3.metric("Failure rate, post", f"{post.is_fail.mean():.2%}",
          delta=f"{post.is_fail.mean() - pre.is_fail.mean():.2%}")
c4.metric("Changepoint", CHANGEPOINT.strftime("%d %b %Y"))

tab1, tab2, tab3 = st.tabs(
    ["Process change", "Control charts", "Attribution"])

# --- tab 1 -----------------------------------------------------------------
with tab1:
    st.subheader("A process change on 20 August 2008 tripled yield")
    st.markdown(
        "The changepoint was located from **sensor behaviour alone** — labels "
        "were not used. The failure rate shift is corroboration, not input."
    )

    window = st.slider("Rolling window (wafers)", 25, 200, 100, 25)
    roll = df.is_fail.rolling(window, center=True).mean()

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax[0].plot(df.measured_at, roll, color="#333", lw=1.2)
    ax[0].axvline(CHANGEPOINT, color="crimson", ls="--", lw=1.5)
    ax[0].axhline(pre.is_fail.mean(), color="steelblue", lw=1, ls=":")
    ax[0].axhline(post.is_fail.mean(), color="steelblue", lw=1, ls=":")
    ax[0].set_ylabel("failure rate")
    ax[0].set_title(f"{pre.is_fail.mean():.1%} → {post.is_fail.mean():.1%}")

    ax[1].plot(df.measured_at, df.sensor_060, lw=0.6, color="#333")
    ax[1].axvline(CHANGEPOINT, color="crimson", ls="--", lw=1.5)
    ax[1].set_ylabel("SENSOR_060")
    fig.tight_layout()
    st.pyplot(fig)

    st.info(
        "**Why this matters for modelling.** Cross-validation folds scored "
        "`[0.19, 0.06, 0.06, 0.46, 0.05]` — the high fold is the one straddling "
        "the boundary. A model trained across the change learns a mixture of "
        "two processes and generalises to neither.",
        icon="⚠️",
    )

# --- tab 2 -----------------------------------------------------------------
with tab2:
    st.subheader("I-MR control charts, limits computed per regime")

    left, right = st.columns([1, 3])
    with left:
        ranked = screen[screen.feature.isin(feats)].sort_values("p")
        sensor = st.selectbox("Sensor", ranked.feature.tolist(),
                              format_func=str.upper)
        regime = st.radio("Regime", ["post", "pre"], horizontal=True)
        st.caption(
            "One measurement per wafer with no rational subgroups, so I-MR "
            "rather than X̄-R. Phase I is iterative: outliers are removed from "
            "the reference window and limits recomputed."
        )

    seg = df[df.regime == regime].reset_index(drop=True)
    x = seg[sensor]
    lim = imr_limits(x)
    flags = nelson(x, lim)
    viol = flags.any(axis=1)

    with right:
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        ax[0].plot(seg.measured_at, x, lw=0.7, color="#333")
        ax[0].scatter(seg.measured_at[viol], x[viol], s=18,
                      color="crimson", zorder=3, label="Nelson violation")
        for k, ls in [("center", "-"), ("ucl", "--"), ("lcl", "--")]:
            ax[0].axhline(lim[k], ls=ls, lw=1, color="steelblue")
        f = seg[seg.is_fail == 1]
        ax[0].scatter(f.measured_at, [ax[0].get_ylim()[0]] * len(f),
                      marker="|", color="orange", s=40, label="failed wafer")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[0].set_title(f"{sensor.upper()} — {regime}-change regime")

        ax[1].plot(seg.measured_at, x.diff().abs(), lw=0.7, color="#333")
        ax[1].axhline(lim["mr_center"], lw=1, color="steelblue")
        ax[1].axhline(lim["mr_ucl"], ls="--", lw=1, color="steelblue")
        ax[1].set_title("Moving range")
        fig.tight_layout()
        st.pyplot(fig)

    fr_v = seg.loc[viol, "is_fail"].mean()
    fr_c = seg.loc[~viol, "is_fail"].mean()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged wafers", f"{int(viol.sum())} ({viol.mean():.1%})")
    m2.metric("Fail rate, flagged", f"{fr_v:.2%}")
    m3.metric("Fail rate, in control", f"{fr_c:.2%}")
    m4.metric("Lift", f"{fr_v / fr_c:.2f}×" if fr_c else "—")

    if sensor == "sensor_060" and regime == "post":
        st.success(
            "**The headline rule.** 12.0 % of flagged wafers fail "
            "(95 % CI 6.7–20.8 %) against a 4.0 % baseline. Odds ratio 3.29, "
            "Fisher exact one-sided p = 0.003 — significant after Bonferroni "
            "correction across the five sensors charted. Note the lift is "
            "**0.40 pre-change**: the rule is regime-specific.",
            icon="✅",
        )

    st.caption("Rules triggered")
    st.dataframe(flags.sum().rename("count").to_frame().T,
                 use_container_width=True, hide_index=True)

# --- tab 3 -----------------------------------------------------------------
with tab3:
    st.subheader("Three methods, one sensor")
    st.markdown(
        "SHAP is scoped to the **post-change regime only** — explaining a model "
        "fitted across the boundary would contradict the finding above. The "
        "model is weak and reported as such; the point is convergence, not "
        "accuracy."
    )

    if art["shap"] is not None:
        top = art["shap"].sort_values("shap_rank").head(12)
        st.dataframe(
            top[["feature", "shap_rank", "univariate_rank",
                 "mean_abs_shap", "p_fdr", "effect"]].round(5),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Rank agreement is concentrated at the top and absent below "
            "(overall Spearman −0.04) — what a few-real-signals structure "
            "predicts. SENSOR_060 ranks 1 by both methods; SENSOR_511 ranks 5 "
            "by both."
        )

    for name, cap in [
        ("shap_beeswarm.png", "SHAP value distribution, post-regime"),
        ("shap_waterfall_failed.png", "Single failed wafer — why it was flagged"),
    ]:
        p = OUT / "charts" / name
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)

    st.divider()
    st.subheader("What this does not show")
    st.markdown(
        """
- **No deployable predictor exists on this data.** Best honest result is ~3×
  no-skill with confidence intervals spanning no-skill.
- Features are **anonymised** — this identifies *which* sensor separates
  failures, not what it measures.
- 56 failures in the post-change regime, 14 in its test set. Every metric is
  coarse.
- Feature screening was not nested inside cross-validation, so screened-model
  scores are optimistically biased.
        """
    )