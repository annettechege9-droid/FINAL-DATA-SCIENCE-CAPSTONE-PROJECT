"""
PowerPulse Kenya - core library
================================
Everything the notebook, the Streamlit app and the slides share lives here, so that
every number in the project comes from ONE implementation.

Design choice: the forecasting models below depend only on numpy, pandas, scipy and
scikit-learn.  SARIMA and Holt-Winters ETS are implemented directly (conditional-sum-of-
squares SARIMA and damped additive Holt-Winters) so the project runs anywhere.  The
notebook additionally cross-checks them against `statsmodels` when it is installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter  # noqa: F401  (kept for users who want to extend)

TARGET = "kplc_sales_gwh"
H_MAX = 12            # forecast horizon: 12 months ahead
M = 12                # seasonal period
FIRST_ORIGIN = 35     # first forecast origin = Dec 2021 (36 months of training data)
DEV_END_IDX = 71      # Dec 2024  -> development / validation window ends here
TEST_START_IDX = 72   # Jan 2025  -> final untouched test window starts here

SOURCE_FILES = {
    "hist_2019_2021": "powerpulse_historical_sales_candidates_2019_2021.csv",
    "hist_2022_2023": "powerpulse_historical_2022_2023_candidates.csv",
    "master_2024_2026": "powerpulse_master_v1_2024-2026.csv",
}

# --------------------------------------------------------------------------------------
# 1. DATA: combine, de-duplicate, audit
# --------------------------------------------------------------------------------------
def build_dataset(data_dir="data"):
    """Concatenate the three non-overlapping source files into one monthly series.

    Priority if a month ever appeared twice: master (3) > 2022-23 file (2) > 2019-21 file (1).
    Returns (dataframe, audit_dict).
    """
    d = Path(data_dir)
    h1 = pd.read_csv(d / SOURCE_FILES["hist_2019_2021"], parse_dates=["date"])
    h2 = pd.read_csv(d / SOURCE_FILES["hist_2022_2023"], parse_dates=["date"])
    mm = pd.read_csv(d / SOURCE_FILES["master_2024_2026"], parse_dates=["date"])

    a = h1[["date", TARGET, "source_report"]].rename(columns={"source_report": "source"})
    a["reliability"] = "candidate_unverified"
    a["origin_file"], a["priority"] = "hist_2019_2021", 1

    b = h2[["date", TARGET, "source"]].copy()
    b["reliability"] = "candidate_unverified"
    b["origin_file"], b["priority"] = "hist_2022_2023", 2

    c = mm[["date", TARGET, "source", "verification_status", "total_generation_gwh",
            "total_imports_gwh", "total_exports_gwh", "total_generation_and_imports_gwh"]].copy()
    c = c.rename(columns={"verification_status": "reliability"})
    c["origin_file"], c["priority"] = "master_2024_2026", 3

    stacked = pd.concat([a, b, c], ignore_index=True)
    n_stacked = len(stacked)
    overlaps = int(stacked["date"].duplicated().sum())
    df = (stacked.sort_values(["date", "priority"], ascending=[True, False])
                 .drop_duplicates("date", keep="first")
                 .sort_values("date").reset_index(drop=True))

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    audit = {
        "rows_stacked": n_stacked,
        "overlapping_dates_resolved": overlaps,
        "rows_final": len(df),
        "start": df["date"].min(), "end": df["date"].max(),
        "missing_months": [str(x.date()) for x in full_range.difference(df["date"])],
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_sales": int(df[TARGET].isna().sum()),
        "non_positive_sales": int((df[TARGET] <= 0).sum()),
        "all_month_start": bool((df["date"].dt.day == 1).all()),
        "unit": "GWh (million kWh)",
    }
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["days_in_month"] = df["date"].dt.days_in_month
    df["sales_per_day_gwh"] = df[TARGET] / df["days_in_month"]
    df["period"] = np.where(df["year"] <= 2023, "2019-2023 (candidate)", "2024-2026 (KNBS verified)")
    return df, audit


def make_split_labels(df):
    """Label every month as train/validation or test according to the project split."""
    idx = np.arange(len(df))
    return np.where(idx <= DEV_END_IDX, "development (train+validation)", "test (held out)")


# --------------------------------------------------------------------------------------
# 2. STATISTICAL TOOLS (no statsmodels needed)
# --------------------------------------------------------------------------------------
def adf_test(x, maxlag=None):
    """Augmented Dickey-Fuller test (constant, AIC lag choice) with MacKinnon critical values."""
    x = np.asarray(x, float)
    n = len(x)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
        maxlag = min(maxlag, n // 2 - 3)
    dx = np.diff(x)

    def design(lags, start):
        y = dx[start:]
        cols = [x[start:-1], np.ones(len(y))]
        for i in range(1, lags + 1):
            cols.append(dx[start - i:len(dx) - i])
        X = np.column_stack(cols)
        return y, X

    best = None
    for lags in range(0, maxlag + 1):                       # compare on a common sample
        y, X = design(lags, maxlag)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        k = X.shape[1]
        aic = len(y) * np.log(resid @ resid / len(y)) + 2 * k
        if best is None or aic < best[0]:
            best = (aic, lags)
    lags = best[1]
    y, X = design(lags, lags)                               # refit on the longest sample
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    t = beta[0] / np.sqrt(cov[0, 0])
    N = len(y)
    crit = {
        "1%": -3.43035 - 6.5393 / N - 16.786 / N**2 - 79.433 / N**3,
        "5%": -2.86154 - 2.8903 / N - 4.234 / N**2 - 40.040 / N**3,
        "10%": -2.56677 - 1.5384 / N - 2.809 / N**2,
    }
    return {"adf_stat": float(t), "lags": int(lags), "nobs": int(N),
            "crit": {k: float(v) for k, v in crit.items()},
            "stationary_at_5pct": bool(t < crit["5%"])}


def classical_decomposition(s, m=M):
    """Additive classical decomposition: centred 2x12 moving-average trend + monthly seasonal."""
    s = pd.Series(s).astype(float)
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    trend = pd.Series(np.convolve(s.values, w, mode="valid"),
                      index=s.index[m // 2: len(s) - m // 2])
    trend = trend.reindex(s.index)
    detr = s - trend
    seas_mean = detr.groupby(s.index.month).mean()
    seas_mean = seas_mean - seas_mean.mean()
    seasonal = pd.Series(s.index.month.map(seas_mean).values, index=s.index)
    remainder = s - trend - seasonal
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "remainder": remainder})


def robust_z(x):
    x = pd.Series(x).astype(float)
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad


def dm_test(e_a, e_b, h=1):
    """Diebold-Mariano test (absolute-error loss, HLN small-sample correction).
    Positive statistic => model A has LARGER errors than model B."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        w = 1 - k / h
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * w * cov
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# --------------------------------------------------------------------------------------
# 3. FORECASTING MODELS  - every model: f(y: pd.Series) -> np.ndarray of length H_MAX
#    `y` is the history available AT THE FORECAST ORIGIN (never any later value).
# --------------------------------------------------------------------------------------
def f_naive(y):
    return np.repeat(y.iloc[-1], H_MAX)


def f_snaive(y):
    v = y.values
    return np.array([v[len(v) - M + h - 1] for h in range(1, H_MAX + 1)])


def f_snaive_drift(y):
    """Seasonal naive, scaled by the trailing 12-month growth rate."""
    v = y.values
    g = v[-12:].sum() / v[-24:-12].sum() - 1
    return f_snaive(y) * (1 + g)


# ---- ETS (Holt-Winters, additive error/seasonal, optional damped trend) ---------------
def _hw_run(yv, alpha, beta_s, gamma_s, phi, l0, b0, s0):
    l, b, s = l0, b0, list(s0)
    sse = 0.0
    for t in range(len(yv)):
        i = t % M
        e = yv[t] - (l + phi * b + s[i])
        sse += e * e
        l = l + phi * b + alpha * e
        b = phi * b + alpha * beta_s * e
        s[i] = s[i] + (1 - alpha) * gamma_s * e
    return sse, l, b, s


def _ets_forecast(y, damped):
    yv = y.values.astype(float)
    n = len(yv)
    l0 = yv[:M].mean()
    b0 = (yv[M:2 * M].mean() - yv[:M].mean()) / M
    s0 = yv[:M] - (l0 + b0 * (np.arange(M) - (M - 1) / 2))
    s0 = s0 - s0.mean()

    def obj(p):
        a, bs, gs = p[:3]
        phi = p[3] if damped else 1.0
        return _hw_run(yv, a, bs, gs, phi, l0, b0, s0)[0]

    bounds = [(0.01, 0.99), (0.0, 1.0), (0.0, 1.0)] + ([(0.80, 0.98)] if damped else [])
    best = None
    for start in ([0.3, 0.1, 0.1, 0.92], [0.6, 0.05, 0.3, 0.9]):
        x0 = start if damped else start[:3]
        r = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    a, bs, gs = best.x[:3]
    phi = best.x[3] if damped else 1.0
    _, l, b, s = _hw_run(yv, a, bs, gs, phi, l0, b0, s0)
    out = []
    for h in range(1, H_MAX + 1):
        trend = b * sum(phi ** i for i in range(1, h + 1))
        out.append(l + trend + s[(n + h - 1) % M])
    return np.array(out), {"alpha": a, "beta_star": bs, "gamma_star": gs, "phi": phi,
                           "level": l, "trend": b, "seasonal": np.array([s[(n + k) % M] for k in range(M)])}


def f_ets_damped(y):
    return _ets_forecast(y, True)[0]


def f_ets_undamped(y):
    return _ets_forecast(y, False)[0]


# ---- SARIMA via conditional sum of squares ---------------------------------------------
def _sarima_polys(params, order, seas):
    p, d, q = order
    P, D, Q = seas
    phi, theta = params[:p], params[p:p + q]
    Phi, Theta = params[p + q:p + q + P], params[p + q + P:p + q + P + Q]
    ar = np.r_[1.0, -phi] if p else np.array([1.0])
    ma = np.r_[1.0, theta] if q else np.array([1.0])
    if P:
        sar = np.zeros(M * P + 1); sar[0] = 1.0
        for j in range(P): sar[M * (j + 1)] = -Phi[j]
        ar = np.convolve(ar, sar)
    if Q:
        sma = np.zeros(M * Q + 1); sma[0] = 1.0
        for j in range(Q): sma[M * (j + 1)] = Theta[j]
        ma = np.convolve(ma, sma)
    return ar, ma


def _css_resid(w, ar, ma):
    n = len(w)
    start = len(ar) - 1
    e = np.zeros(n)
    for t in range(start, n):
        val = w[t]
        for k in range(1, len(ar)):
            val += ar[k] * w[t - k]
        for k in range(1, len(ma)):
            if t - k >= start:
                val -= ma[k] * e[t - k]
        e[t] = val
    return e, start


def _sarima_forecast(y, order, seas):
    z = y.values.astype(float)
    p, d, q = order
    P, D, Q = seas
    w = z.copy()
    for _ in range(D): w = w[M:] - w[:-M]
    for _ in range(d): w = np.diff(w)
    k = p + q + P + Q

    def sse(params):
        ar, ma = _sarima_polys(params, order, seas)
        e, st = _css_resid(w, ar, ma)
        return float(e[st:] @ e[st:])

    r = minimize(sse, np.full(k, 0.1), bounds=[(-0.98, 0.98)] * k, method="L-BFGS-B")
    ar, ma = _sarima_polys(r.x, order, seas)
    e, start = _css_resid(w, ar, ma)
    # forecast w with future shocks = 0
    wf, ef = list(w), list(e)
    for _ in range(H_MAX):
        t = len(wf)
        val = 0.0
        for kk in range(1, len(ar)):
            val -= ar[kk] * wf[t - kk]
        for kk in range(1, len(ma)):
            if t - kk < len(ef):
                val += ma[kk] * ef[t - kk]
        wf.append(val); ef.append(0.0)
    w_hat = np.array(wf[len(w):])
    # un-difference
    diffpoly = np.array([1.0])
    for _ in range(d): diffpoly = np.convolve(diffpoly, [1.0, -1.0])
    for _ in range(D):
        s = np.zeros(M + 1); s[0] = 1.0; s[M] = -1.0
        diffpoly = np.convolve(diffpoly, s)
    zf = list(z)
    for h in range(H_MAX):
        t = len(zf)
        val = w_hat[h]
        for kk in range(1, len(diffpoly)):
            val -= diffpoly[kk] * zf[t - kk]
        zf.append(val)
    return np.array(zf[len(z):])


SARIMA_ORDERS = {
    "SARIMA_A": ((0, 1, 1), (0, 1, 1)),   # the classic "airline" model
    "SARIMA_B": ((1, 1, 1), (0, 1, 1)),
    "SARIMA_C": ((1, 1, 0), (0, 1, 1)),
}


def _mk_sarima(name):
    order, seas = SARIMA_ORDERS[name]
    return lambda y: _sarima_forecast(y, order, seas)


# ---- Linear regression with trend + calendar -------------------------------------------
def _cal_features(dates, t0):
    t = ((dates.year - t0.year) * 12 + (dates.month - t0.month)).values.astype(float)
    X = np.column_stack([t] + [(dates.month == m).astype(float) for m in range(2, 13)]
                        + [(dates.is_leap_year & (dates.month == 2)).astype(float)])
    return X


def f_linreg(y):
    from sklearn.linear_model import LinearRegression
    fut = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H_MAX, freq="MS")
    t0 = y.index[0]
    mdl = LinearRegression().fit(_cal_features(y.index, t0), y.values)
    return mdl.predict(_cal_features(fut, t0))


# ---- Direct multi-horizon ML on year-on-year growth of DAILY-AVERAGE sales -------------
# Working with sales-per-day removes the calendar (days-in-month / leap-year) effect, so the
# learners only need to predict genuine year-on-year growth.  The forecast is rebuilt as
#     sales(t) = sales_per_day(t-12) * exp(predicted log growth) * days_in_month(t)
def _growth_features(u, o):
    g = lambda k: u[k] / u[k - M] - 1
    g0, g1, g2 = g(o), g(o - 1), g(o - 2)
    m6 = np.mean([g(o - k) for k in range(6)])
    mom = np.mean([g(o - k) for k in range(3)]) - np.mean([g(o - k) for k in range(3, 6)])
    return [g0, g1, g2, m6, mom]


def _target_features(dates, t, h):
    d = dates[t]
    return [h, np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)]


FEATURE_NAMES = ["yoy_growth_t0", "yoy_growth_t-1", "yoy_growth_t-2", "yoy_growth_mean_6m",
                 "yoy_momentum", "horizon", "month_sin", "month_cos"]


def _covid_contaminated_growth(dates):
    """YoY growth at index k is distorted if month k or month k-12 lies in Apr-Jun 2020
    (the COVID shock and its Apr-Jun 2021 'rebound').  Those growth values are not a
    repeatable seasonal pattern, so they are masked from ML TRAINING (never from evaluation)."""
    shock = (dates >= pd.Timestamp("2020-04-01")) & (dates <= pd.Timestamp("2020-06-01"))
    bad = np.zeros(len(dates), bool)
    for k in range(len(dates)):
        bad[k] = shock[k] or (k >= M and shock[k - M])
    return bad


def _direct_xy(u, dates, H=H_MAX, return_t=False):
    """Training rows: (origin o, horizon h) with target index o+h <= len(u)-1  (no look-ahead).
    Rows whose features or target rely on COVID-distorted growth are skipped."""
    X, Y, T = [], [], []
    n = len(u)
    bad = _covid_contaminated_growth(dates)
    for o in range(M + 5, n - 1):
        if bad[o - 5:o + 1].any():
            continue
        base = _growth_features(u, o)
        for h in range(1, H + 1):
            t = o + h
            if t > n - 1:
                break
            if bad[t]:
                continue
            X.append(base + _target_features(dates, t, h))
            Y.append(np.log(u[t] / u[t - M]))
            T.append(t)
    if return_t:
        return np.array(X), np.array(Y), np.array(T)
    return np.array(X), np.array(Y)


def _direct_fit(y, make_model, seed=0):
    u = (y / y.index.days_in_month).values.astype(float)
    fut = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H_MAX, freq="MS")
    dates = y.index.append(fut)
    X, Y = _direct_xy(u, dates)
    return make_model(seed).fit(X, Y), u, dates


def _direct_forecast(y, make_model, n_seeds=1):
    n = len(y)
    o = n - 1
    preds = []
    for s in range(n_seeds):
        mdl, u, dates = _direct_fit(y, make_model, s)
        base = _growth_features(u, o)
        Xf = np.array([base + _target_features(dates, o + h, h) for h in range(1, H_MAX + 1)])
        preds.append(mdl.predict(Xf))
    gp = np.mean(preds, axis=0)
    return np.array([u[o + h - M] * np.exp(gp[h - 1]) * dates[o + h].days_in_month
                     for h in range(1, H_MAX + 1)])


def _rf(depth):
    from sklearn.ensemble import RandomForestRegressor
    return lambda s: RandomForestRegressor(n_estimators=150, max_depth=depth, min_samples_leaf=3,
                                           random_state=42 + s, n_jobs=1)


def f_gbr(y):
    from sklearn.ensemble import GradientBoostingRegressor
    return _direct_forecast(y, lambda s: GradientBoostingRegressor(
        n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42 + s))


def f_mlp(y):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _direct_forecast(y, lambda s: make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=30.0, activation="tanh",
                                           max_iter=3000, random_state=42 + s)), n_seeds=3)


def candidate_models():
    """All candidate configurations. Families with several configs are resolved on DEVELOPMENT data."""
    return {
        "Naive": f_naive,
        "Seasonal naive": f_snaive,
        "Seasonal naive + drift": f_snaive_drift,
        "ETS_damped": f_ets_damped,
        "ETS_undamped": f_ets_undamped,
        "SARIMA_A": _mk_sarima("SARIMA_A"),
        "SARIMA_B": _mk_sarima("SARIMA_B"),
        "SARIMA_C": _mk_sarima("SARIMA_C"),
        "Linear regression": f_linreg,
        "RF_depth3": lambda y: _direct_forecast(y, _rf(3)),
        "RF_depthNone": lambda y: _direct_forecast(y, _rf(None)),
        "Gradient boosting": f_gbr,
        "Neural net (MLP)": f_mlp,
    }


FAMILIES = {"ETS": ["ETS_damped", "ETS_undamped"],
            "SARIMA": ["SARIMA_A", "SARIMA_B", "SARIMA_C"],
            "Random forest": ["RF_depth3", "RF_depthNone"]}


# --------------------------------------------------------------------------------------
# 4. ROLLING-ORIGIN EVALUATION
# --------------------------------------------------------------------------------------
def rolling_origin(y, models, first_origin=FIRST_ORIGIN, verbose=True):
    """Expanding-window rolling-origin forecasts.

    At each origin o the model sees ONLY y[0..o] and forecasts months o+1 .. o+12.
    Returns a long dataframe: model, origin_date, h, target_date, forecast, actual.
    """
    rows = []
    n = len(y)
    for o in range(first_origin, n - 1):
        hist = y.iloc[: o + 1]
        for name, fn in models.items():
            f = fn(hist)
            if not np.all(np.isfinite(f)):
                raise ValueError(f"non-finite forecast from {name} at origin {y.index[o]}")
            for h in range(1, H_MAX + 1):
                t = o + h
                if t > n - 1:
                    break
                rows.append((name, o, y.index[o], h, t, y.index[t], f[h - 1], y.iloc[t]))
        if verbose and (o - first_origin) % 5 == 0:
            print(f"  origin {y.index[o].date()} done", flush=True)
    out = pd.DataFrame(rows, columns=["model", "origin_idx", "origin_date", "h", "target_idx",
                                      "target_date", "forecast", "actual"])
    out["error"] = out["actual"] - out["forecast"]
    out["window"] = np.where(out["target_idx"] <= DEV_END_IDX, "development", "test")
    return out


def add_ensemble(res, members, name="Ensemble (ETS+SARIMA+LinReg+RF)"):
    sub = res[res["model"].isin(members)]
    keys = ["origin_idx", "origin_date", "h", "target_idx", "target_date", "actual", "window"]
    ens = sub.groupby(keys, as_index=False)["forecast"].mean()
    ens["model"] = name
    ens["error"] = ens["actual"] - ens["forecast"]
    return pd.concat([res, ens[res.columns]], ignore_index=True)


def metrics(df, by=("model",), scale=None):
    g = df.assign(abs_err=lambda d: d["error"].abs(), sq=lambda d: d["error"] ** 2,
                  ape=lambda d: d["error"].abs() / d["actual"] * 100).groupby(list(by))
    out = g.agg(MAE=("abs_err", "mean"), RMSE=("sq", lambda s: np.sqrt(s.mean())),
                MAPE=("ape", "mean"), Bias=("error", "mean"), n=("error", "size")).reset_index()
    if scale:
        out["MASE"] = out["MAE"] / scale
    return out


def mase_scale(y, upto=60):
    v = y.values[:upto]
    return float(np.mean(np.abs(v[M:] - v[:-M])))


def select_configs(res):
    """Pick one configuration per family using DEVELOPMENT-window MAE averaged over horizons 1-12."""
    dev = res[res["window"] == "development"]
    per_h = dev.groupby(["model", "h"])["error"].apply(lambda e: e.abs().mean()).reset_index()
    score = per_h.groupby("model")["error"].mean()
    chosen = {fam: min(cfgs, key=lambda c: score[c]) for fam, cfgs in FAMILIES.items()}
    return chosen, score.sort_values()


def finalize_names(res, chosen):
    """Rename chosen configs to family names and drop the losing configs."""
    keep = set(chosen.values()) | {"Naive", "Seasonal naive", "Seasonal naive + drift",
                                   "Linear regression", "Gradient boosting", "Neural net (MLP)"}
    out = res[res["model"].isin(keep)].copy()
    out["model"] = out["model"].replace({v: k for k, v in chosen.items()})
    return out


def interval_table(res_model, coverage=0.8):
    """Empirical symmetric prediction intervals from rolling-origin absolute errors, per horizon."""
    q = res_model.groupby("h")["error"].apply(lambda e: np.quantile(np.abs(e), coverage))
    return q


def interval_coverage(calib, evald, coverage=0.8):
    q = interval_table(calib, coverage)
    e = evald.copy()
    e["q"] = e["h"].map(q)
    e["inside"] = e["error"].abs() <= e["q"]
    return e.groupby("h")["inside"].mean()
