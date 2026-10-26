"""
PowerPulse Kenya - interactive dashboard
Run with:   streamlit run app.py
Reads the files produced by PowerPulse_Capstone.ipynb (folders `data/` and `outputs/`).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
OUT, DATA = BASE / "outputs", BASE / "data"
TARGET = "kplc_sales_gwh"
ENS = "Ensemble (ETS+SARIMA+LinReg+RF)"
NAVY, TEAL, AMBER, RED, GREY = "#12355B", "#1B998B", "#F2A541", "#C0392B", "#8A9BA8"

st.set_page_config(page_title="PowerPulse Kenya", page_icon="⚡", layout="wide")


# ------------------------------------------------------------------ data loading
@st.cache_data
def load():
    need = ["rolling_origin_forecasts.csv", "model_metrics_by_horizon.csv", "final_forecast_next_12_months.csv"]
    missing = [f for f in need if not (OUT / f).exists()] + ([] if (DATA / "powerpulse_final_dataset.csv").exists() else ["data/powerpulse_final_dataset.csv"])
    if missing:
        return None, missing
    d = {
        "df": pd.read_csv(DATA / "powerpulse_final_dataset.csv", parse_dates=["date"]),
        "ro": pd.read_csv(OUT / "rolling_origin_forecasts.csv", parse_dates=["origin_date", "target_date"]),
        "met": pd.read_csv(OUT / "model_metrics_by_horizon.csv"),
        "fc": pd.read_csv(OUT / "final_forecast_next_12_months.csv", parse_dates=["date"]),
        "summ": pd.read_csv(OUT / "model_metrics_summary.csv"),
        "triage": pd.read_csv(OUT / "file_triage.csv"),
    }
    for name, key in [("rf_feature_importance.csv", "rf"), ("linreg_coefficients.csv", "lin"),
                      ("interval_calibration.csv", "cal"), ("sensitivity_may_june_2025.csv", "sens"),
                      ("decomposition.csv", "dec"), ("diebold_mariano.csv", "dm")]:
        p = OUT / name
        d[key] = pd.read_csv(p) if p.exists() else None
    return d, []


D, missing = load()
if D is None:
    st.error("Results not found. Run **PowerPulse_Capstone.ipynb** first. Missing: " + ", ".join(missing))
    st.stop()
df, ro, met, fc = D["df"], D["ro"], D["met"], D["fc"]
MODELS = sorted(ro["model"].unique(), key=lambda m: (m != ENS, m))

# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ PowerPulse Kenya")
st.sidebar.caption("Forecasting monthly KPLC electricity sales (GWh)\n\nFinal capstone project - Annette Chege")
PAGES = ["Overview", "Data & quality", "Explore history", "Model comparison",
         "Forecast explorer", "Next 12 months", "Explainability", "Method & limitations"]
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Data: KNBS Leading Economic Indicators, Table 15(b). 2019-2023 values are *candidates* awaiting verification; "
                   "2024-2026 values carry KNBS verification notes; Jan-Mar 2026 are provisional.")


def show(fig):
    st.pyplot(fig, clear_figure=True)


# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("PowerPulse Kenya")
    st.subheader("Forecasting monthly electricity sales with time series analysis and machine learning")
    test = D["summ"][D["summ"]["window"] == "test"].sort_values("MAE")
    valid = D["summ"][D["summ"]["window"] == "validation"].sort_values("MAE")
    last12 = df[TARGET].iloc[-12:].sum()
    next12 = fc[ENS].sum()
    c = st.columns(4)
    c[0].metric("Months of data", f"{len(df)}", f"{df['date'].min():%b %Y} - {df['date'].max():%b %Y}", delta_color="off")
    c[1].metric("Latest month (provisional)", f"{df[TARGET].iloc[-1]:,.0f} GWh", f"{df['date'].iloc[-1]:%b %Y}", delta_color="off")
    c[2].metric("Next 12 months (forecast)", f"{next12:,.0f} GWh", f"{(next12 / last12 - 1) * 100:+.1f}% vs last 12 months")
    c[3].metric("Ensemble error, validation / test", f"{valid.set_index('model').loc[ENS, 'MAPE']:.1f}% / {test.set_index('model').loc[ENS, 'MAPE']:.1f}%",
                "MAPE averaged over 1-12 months ahead", delta_color="off")
    st.markdown("""
**Key messages**
1. **Sales are growing** - average monthly sales rose from about 739 GWh (2019) to about 982 GWh (2025), with a COVID-19 dip in 2020 and an unusually strong 2025.
2. **Strong seasonal pattern** - high in January, June, August and October; low in February, April and December.
3. **Simple models are hard to beat.** Combining ETS, SARIMA, linear regression and random forest gives the best validation accuracy; no complex model is *significantly* better than a seasonal-naive-with-drift benchmark.
4. **Be careful with May-June 2025.** Sales jumped while electricity supply did not - this looks like a reporting effect, and it explains much of the 2025-26 forecast error.
5. **Forecast intervals are optimistic** - in the test period, only about 42% of months fell inside the nominal 80% interval.
""")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(df["date"], df[TARGET], color=NAVY, lw=1.8, label="Actual")
    ax.fill_between(fc["date"], fc["lower_80"], fc["upper_80"], color=TEAL, alpha=.3, label="80% interval")
    ax.plot(fc["date"], fc[ENS], color=RED, lw=2.2, marker="o", ms=3, label="Ensemble forecast")
    ax.set_ylabel("GWh"); ax.legend(); ax.set_title("History and 12-month forecast")
    show(fig)

elif page == "Data & quality":
    st.header("Data & quality")
    st.markdown("**87 monthly observations** were assembled from three non-overlapping source files. Four other uploaded files were audit outputs or duplicates and were deliberately **not** stacked.")
    st.subheader("Which files were combined?")
    st.dataframe(D["triage"], hide_index=True)
    st.subheader("Reliability of each month")
    rel = df.groupby(["period", "reliability"]).agg(months=("date", "size"), first=("date", "min"), last=("date", "max")).reset_index()
    st.dataframe(rel, hide_index=True)
    colmap = {"candidate_unverified": GREY, "verified": NAVY, "verified_target_2src": NAVY, "verified_but_anomalous": RED, "verified_provisional": AMBER}
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df["date"], df[TARGET], color="#c5d0d8", lw=1)
    for r, c in colmap.items():
        s = df[df["reliability"] == r]
        ax.scatter(s["date"], s[TARGET], color=c, s=22, label=r.replace("_", " "))
    ax.legend(fontsize=7, ncol=3); ax.set_ylabel("GWh")
    show(fig)
    st.subheader("Final dataset")
    st.dataframe(df[["date", TARGET, "days_in_month", "sales_per_day_gwh", "reliability", "source"]], hide_index=True)
    st.download_button("Download final dataset (CSV)", df.to_csv(index=False).encode(), "powerpulse_final_dataset.csv")
    st.info("Honest note: the 60 months from 2019-2023 are collected but not independently verified; the verification register still marks them 'needs_source_check'.")

elif page == "Explore history":
    st.header("Explore historical sales")
    lo, hi = df["date"].min().to_pydatetime(), df["date"].max().to_pydatetime()
    rng = st.slider("Date range", lo, hi, (lo, hi), format="MMM YYYY")
    view = st.radio("Measure", ["Monthly sales (GWh)", "Average daily sales (GWh per day)", "Year-on-year change (%)"], horizontal=True)
    s = df[(df["date"] >= rng[0]) & (df["date"] <= rng[1])].copy()
    s["yoy"] = df[TARGET].pct_change(12).mul(100).reindex(s.index)
    fig, ax = plt.subplots(figsize=(11, 4))
    if view.startswith("Monthly"):
        ax.plot(s["date"], s[TARGET], color=NAVY, marker="o", ms=3); ax.set_ylabel("GWh")
    elif view.startswith("Average"):
        ax.plot(s["date"], s["sales_per_day_gwh"], color=TEAL, marker="o", ms=3); ax.set_ylabel("GWh per day")
    else:
        ax.bar(s["date"], s["yoy"], width=25, color=[RED if v < 0 else TEAL for v in s["yoy"].fillna(0)]); ax.set_ylabel("%"); ax.axhline(0, color="k", lw=.7)
    show(fig)
    a, b = st.columns(2)
    with a:
        st.subheader("Seasonality heat map")
        piv = s.pivot(index="year", columns="month_num", values=TARGET)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
        plt.colorbar(im, ax=ax, label="GWh"); show(fig)
    with b:
        st.subheader("Annual averages")
        ann = s.groupby("year")[TARGET].agg(["mean", "min", "max", "count"]).round(1)
        st.dataframe(ann)
        st.caption("A year with fewer than 12 months (2026) is not comparable with full years.")
    if D["dec"] is not None:
        st.subheader("Trend, seasonal and remainder")
        dec = D["dec"].copy(); dec["date"] = pd.to_datetime(dec["date"])
        fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
        for ax, col, c in zip(axes, ["trend", "seasonal", "remainder"], [TEAL, AMBER, RED]):
            ax.plot(dec["date"], dec[col], color=c); ax.set_ylabel(col)
        show(fig)

elif page == "Model comparison":
    st.header("Model comparison")
    st.caption("Validation = forecast targets Jan 2022 - Dec 2024 (used to choose configurations). Test = Jan 2025 - Mar 2026 (never used for any choice).")
    c1, c2, c3 = st.columns(3)
    win = c1.selectbox("Window", ["development", "test"], format_func=lambda x: "Validation" if x == "development" else "Test (held out)")
    metric = c2.selectbox("Metric", ["MAE", "RMSE", "MAPE", "MASE"])
    hz = c3.selectbox("Forecast horizon", ["All horizons (1-12)", 1, 3, 6, 12])
    chosen = st.multiselect("Models", MODELS, default=MODELS)
    m = met[(met["window"] == win) & met["model"].isin(chosen)]
    if hz != "All horizons (1-12)":
        m = m[m["h"] == hz]
    agg = m.groupby("model")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(agg.index, agg.values, color=[RED if i == ENS else TEAL for i in agg.index]); ax.invert_yaxis()
    ax.set_xlabel(metric + (" (GWh)" if metric in ("MAE", "RMSE") else (" (%)" if metric == "MAPE" else "")))
    for i, v in enumerate(agg.values): ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    show(fig)
    st.subheader(f"{metric} by forecast horizon")
    pv = met[(met["window"] == win) & met["model"].isin(chosen)].pivot(index="h", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in pv.columns: ax.plot(pv.index, pv[c], marker="o", ms=3, lw=2.6 if c == ENS else 1.1, label=c)
    ax.set_xlabel("Months ahead"); ax.set_ylabel(metric); ax.legend(fontsize=7, ncol=2)
    show(fig)
    st.dataframe(pv.round(2))
    if D["dm"] is not None:
        with st.expander("Is the improvement statistically significant? (Diebold-Mariano tests)"):
            dm = D["dm"]
            base = st.selectbox("Compare against", ["Seasonal naive + drift", "Seasonal naive"])
            t = dm[dm["vs"] == base].copy()
            t["result"] = t["MAE_diff_GWh"].round(1).astype(str) + " GWh (p=" + t["p_value"].round(3).astype(str) + ")"
            st.dataframe(t.pivot(index="model", columns="h", values="result"))
            st.caption("Negative = model more accurate than the baseline. p < 0.05 means the difference is unlikely to be luck.")
    if D["sens"] is not None:
        with st.expander("Sensitivity: how much do May-June 2025 matter?"):
            st.dataframe(D["sens"], hide_index=True)

elif page == "Forecast explorer":
    st.header("Forecast explorer - what would each model have predicted?")
    st.caption("Pick any past forecast origin: every model sees only data up to that month and forecasts the following months.")
    origins = sorted(ro["origin_date"].unique())
    o = st.select_slider("Forecast origin (last month of data the models can see)", options=origins, value=origins[len(origins) // 2],
                         format_func=lambda x: pd.Timestamp(x).strftime("%b %Y"))
    sel = st.multiselect("Models", MODELS, default=[ENS, "SARIMA", "Seasonal naive + drift", "Naive"])
    hmax = st.slider("Months ahead to show", 1, 12, 12)
    s = ro[(ro["origin_date"] == o) & (ro["h"] <= hmax) & ro["model"].isin(sel)]
    hist = df[(df["date"] <= pd.Timestamp(o) + pd.DateOffset(months=hmax + 1)) & (df["date"] >= pd.Timestamp(o) - pd.DateOffset(months=24))]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(hist["date"], hist[TARGET], color="k", lw=2.2, label="Actual")
    for mname, g in s.groupby("model"):
        ax.plot(g["target_date"], g["forecast"], lw=2.6 if mname == ENS else 1.3, marker="o", ms=3, label=mname)
    ax.axvline(pd.Timestamp(o), color=GREY, ls=":"); ax.set_ylabel("GWh"); ax.legend(fontsize=8)
    show(fig)
    t = s.assign(abs_error=lambda x: x["error"].abs()).groupby("model")["abs_error"].mean().round(1).rename("mean |error| (GWh)")
    st.dataframe(t.sort_values())
    st.dataframe(s.pivot(index="target_date", columns="model", values="forecast").round(1))

elif page == "Next 12 months":
    st.header("Forecast: April 2026 - March 2027")
    lvl = st.radio("Prediction interval", ["80%", "90%"], horizontal=True)
    scen = st.checkbox("Show scenario: May-June 2025 were one-offs", value=True)
    k = lvl[:2]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(df["date"].iloc[-36:], df[TARGET].iloc[-36:], color=NAVY, lw=2, label="Actual")
    ax.fill_between(fc["date"], fc[f"lower_{k}"], fc[f"upper_{k}"], color=TEAL, alpha=.28, label=f"{lvl} interval")
    ax.plot(fc["date"], fc[ENS], color=RED, lw=2.4, marker="o", ms=4, label="Ensemble forecast")
    sc = "ensemble_if_MayJun2025_are_one_offs"
    if scen and sc in fc:
        ax.plot(fc["date"], fc[sc], color=AMBER, ls="--", lw=2, label="Scenario: May/Jun-2025 one-offs")
    ax.set_ylabel("GWh"); ax.legend()
    show(fig)
    tbl = fc[["date", ENS, f"lower_{k}", f"upper_{k}", "same_month_last_year", "implied_yoy_%"] + ([sc] if sc in fc else [])].copy()
    tbl["date"] = tbl["date"].dt.strftime("%b %Y")
    st.dataframe(tbl.round(1), hide_index=True)
    last12 = df[TARGET].iloc[-12:].sum()
    st.metric("12-month total", f"{fc[ENS].sum():,.0f} GWh", f"{(fc[ENS].sum() / last12 - 1) * 100:+.1f}% vs last 12 months ({last12:,.0f} GWh)")
    st.download_button("Download forecast (CSV)", fc.to_csv(index=False).encode(), "powerpulse_forecast.csv")
    st.warning("Intervals are empirical (from past rolling-origin errors) and were too narrow during 2025-26: only ~42% of test months fell inside the nominal 80% band. "
               "Treat them as optimistic. Mar-2026 data are provisional.")
    st.subheader("Individual model forecasts")
    indiv = fc.drop(columns=[c for c in fc.columns if c.startswith(("lower", "upper"))]).copy()
    indiv["date"] = indiv["date"].dt.strftime("%b %Y")
    st.dataframe(indiv.round(1), hide_index=True)

elif page == "Explainability":
    st.header("Explaining the models")
    a, b = st.columns(2)
    with a:
        st.subheader("Linear regression: effect of each month (GWh)")
        if D["lin"] is not None:
            lin = D["lin"]
            st.write(f"**Trend: {lin.set_index('term').loc['trend (GWh per month)', 'coefficient']:+.2f} GWh per month.** Other terms show how each month differs from January.")
            c = lin[lin["term"].str.contains(" vs Jan")]
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.bar(c["term"].str.replace(" vs Jan", ""), c["coefficient"], color=[TEAL if v >= 0 else RED for v in c["coefficient"]])
            ax.axhline(0, color="k", lw=.7); show(fig)
    with b:
        st.subheader("Random forest: feature importance")
        if D["rf"] is not None:
            rf = D["rf"].sort_values("impurity_importance")
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.barh(rf["feature"], rf["impurity_importance"], color=TEAL); show(fig)
            st.caption("The forest mostly extrapolates recent year-on-year growth and adjusts it slightly for the calendar month.")
    st.subheader("Forecast uncertainty")
    if D["cal"] is not None:
        st.dataframe(D["cal"].round(2), hide_index=True)
        st.caption("Half-width of an 80% interval calibrated on validation errors, and how often the test months actually fell inside it.")
    st.subheader("Actual vs predicted")
    mod = st.selectbox("Model", MODELS)
    hh = st.select_slider("Horizon (months ahead)", options=list(range(1, 13)), value=3)
    s = ro[(ro["model"] == mod) & (ro["h"] == hh)].sort_values("target_date")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["date"], df[TARGET], color="#c5d0d8"); axes[0].plot(s["target_date"], s["forecast"], color=RED)
    axes[0].axvline(pd.Timestamp("2025-01-01"), color=AMBER, ls="--"); axes[0].set_title("Actual (grey) vs forecast (red); dashed = start of test")
    axes[1].scatter(s["actual"], s["forecast"], c=np.where(s["window"] == "test", RED, TEAL), s=18)
    lims = [s["actual"].min() - 10, s["actual"].max() + 10]; axes[1].plot(lims, lims, "k--", lw=.8)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Forecast"); axes[1].set_title("Predicted vs actual (teal = validation, red = test)")
    show(fig)
    for f in ["10_error_heatmap.png", "08_actual_vs_predicted_all.png"]:
        p = OUT / "figs" / f
        if p.exists():
            st.image(str(p), caption=f.replace("_", " ").replace(".png", ""))

else:
    st.header("Method & limitations")
    st.markdown("""
**Target.** Monthly KPLC electricity *sales* in GWh (KNBS Table 15(b)). This is neither total consumption nor peak demand.

**Forecast objective.** 12 months ahead (also reported at 1, 3, 6 months) - budgeting, procurement and planning.

**Evaluation.** Chronological splits; expanding-window rolling origin. Initial training Jan 2019 - Dec 2021; validation targets Jan 2022 - Dec 2024 (all choices made here); test targets Jan 2025 - Mar 2026 (held out). Every model sees only data available at the forecast origin.

**Models compared.** Naive, seasonal naive, seasonal naive + drift (baselines); ETS; SARIMA; linear regression; random forest; gradient boosting; neural net (MLP); and an ensemble of ETS, SARIMA, linear regression and random forest.

**Machine-learning inputs.** Year-on-year growth of daily-average sales (latest, previous two months, 6-month mean, momentum), horizon, and month of year. COVID-distorted growth values are excluded from training.

**Limitations.**
- Only 87 months; 60 are unverified candidates from 2019-2023.
- May-June 2025 are unexplained and drive much of the test error (see sensitivity analysis).
- Short test window containing a growth acceleration; intervals under-cover.
- No external drivers (GDP, weather, tariffs); no LSTM (too little data; listed as future work).
""")
