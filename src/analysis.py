"""Every regression in the project. Each call to `fit` is logged to results/spec_registry.csv,
so the number of specifications tried is on the record rather than in the author's head.

Run after build_panel.py:  python src/analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel_monthly.csv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CHANNELS = ["ch_financial", "ch_real"]
CATEGORIES = ["cat_securities", "cat_intermediation", "cat_government", "cat_war", "cat_disaster"]
REGISTRY: list[dict] = []


def nw_lags(h: int) -> int:
    """Newey-West lag length: at least 6 months, and at least twice the overlap for multi-month targets."""
    return max(6, 2 * h)


def fit(df: pd.DataFrame, y: str, x: list[str], h: int, family: str, label: str, lags: int | None = None) -> dict:
    d = df[[y] + x].dropna()
    lags = nw_lags(h) if lags is None else lags
    res = sm.OLS(d[y], sm.add_constant(d[x])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    row = {"family": family, "label": label, "y": y, "h": h, "n": int(res.nobs),
           "start": d.index.min(), "end": d.index.max(), "r2": res.rsquared, "nw_lags": lags}
    for v in x:
        row[f"b_{v}"] = res.params[v]
        row[f"t_{v}"] = res.tvalues[v]
        row[f"p_{v}"] = res.pvalues[v]
    REGISTRY.append(row)
    return row


def load() -> pd.DataFrame:
    p = pd.read_csv(PANEL, index_col=0)
    # Standardise news regressors on the full sample so coefficients read "per one standard deviation".
    for c in CHANNELS + CATEGORIES + ["nvix"]:
        p[c + "_z"] = (p[c] - p[c].mean()) / p[c].std()
    # HAR-style realised volatility controls (Corsi 2009), in logs.
    p["lrv_1"] = np.log(p["rv"]) / 2
    p["lrv_3"] = np.log(p["rv"].rolling(3).mean()) / 2
    p["lrv_12"] = np.log(p["rv"].rolling(12).mean()) / 2
    for h in (1, 3, 6, 12):
        # Target: log realised volatility over the next h months (average variance, then sqrt).
        fwd = p["rv"].rolling(h).mean().shift(-h)
        p[f"y_lrv_{h}"] = np.log(fwd) / 2
        # Target: change in log implied volatility over the next h months.
        p[f"y_dliv_{h}"] = p["liv"].shift(-h) - p["liv"]
    p["y_vrp_1"] = p["vrp"].shift(-1)
    return p


def sub(p: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    return p[p["window"].isin(windows)]


HAR = ["lrv_1", "lrv_3", "lrv_12"]
CH = [c + "_z" for c in CHANNELS]
CAT = [c + "_z" for c in CATEGORIES]
CLEAN_IV = ["svr_test", "post_sample"]  # windows where NVIX was not fitted to implied vol


def table_quantity(p):
    """Q1. Does news predict the quantity of risk, i.e. realised volatility, beyond its own history?"""
    rows = []
    samples = {
        "1926-1985 (before VIX; text model never saw it)": ["pre_vix"],
        "1986-1995 + 2010-2016 (clean, adds implied vol)": CLEAN_IV,
        "1996-2009 (text model trained here)": ["svr_train"],
    }
    for sname, w in samples.items():
        d = sub(p, w)
        ctrl = HAR + (["liv"] if w != ["pre_vix"] else [])
        for h in (1, 3, 6, 12):
            rows.append(fit(d, f"y_lrv_{h}", CH + ctrl, h, "quantity", sname))
    return pd.DataFrame(rows)


def table_implied(p):
    """Q2. The original test, done properly: next-month change in implied vol, by estimation window."""
    rows = []
    for sname, w in {"1986-1995 (text model hold-out)": ["svr_test"],
                     "2010-2016 (post-sample)": ["post_sample"],
                     "1986-1995 + 2010-2016 pooled (clean)": CLEAN_IV,
                     "1996-2009 (text model trained here)": ["svr_train"]}.items():
        d = sub(p, w)
        rows.append(fit(d, "y_dliv_1", CH + ["liv"], 1, "implied", sname))
        rows.append(fit(d, "y_dliv_1", CH + ["liv"] + HAR, 1, "implied_har", sname))
    return pd.DataFrame(rows)


def table_price(p):
    """Q3. Does news move the price of risk (variance risk premium) rather than the quantity?"""
    rows = []
    for sname, w in {"1986-1995 + 2010-2016 pooled (clean)": CLEAN_IV,
                     "1996-2009 (text model trained here)": ["svr_train"]}.items():
        d = sub(p, w)
        rows.append(fit(d, "y_vrp_1", CH + ["vrp", "liv"], 1, "price", sname))
    return pd.DataFrame(rows)


def table_categories(p):
    """Q4. Which single category carries the financial channel? Five categories, Holm-adjusted."""
    rows = []
    d = sub(p, ["pre_vix"])
    for h in (1, 3, 12):
        rows.append(fit(d, f"y_lrv_{h}", CAT + HAR, h, "categories", "1926-1985"))
    out = pd.DataFrame(rows)
    pcols = [f"p_{c}" for c in CAT]
    flat = out[pcols].to_numpy().ravel()
    adj = multipletests(flat, method="holm")[1].reshape(out[pcols].shape)
    for j, c in enumerate(pcols):
        out[c.replace("p_", "pholm_")] = adj[:, j]
    return out


def table_break(p):
    """Q5. Did the financial channel strengthen after 2008? Tested only on data the text model never fitted."""
    rows = []
    # Realised-vol target: compare 1926-1985 with 2010-2016, both outside training.
    d = sub(p, ["pre_vix", "post_sample"]).copy()
    d["post"] = (d["window"] == "post_sample").astype(float)
    d["fin_x_post"] = d["ch_financial_z"] * d["post"]
    d["real_x_post"] = d["ch_real_z"] * d["post"]
    rows.append(fit(d, "y_lrv_1", CH + ["post", "fin_x_post", "real_x_post"] + HAR, 1, "break",
                    "1926-1985 vs 2010-2016, realised vol"))
    # Implied-vol target: 1986-1995 against 2010-2016.
    d = sub(p, CLEAN_IV).copy()
    d["post"] = (d["window"] == "post_sample").astype(float)
    d["fin_x_post"] = d["ch_financial_z"] * d["post"]
    d["real_x_post"] = d["ch_real_z"] * d["post"]
    rows.append(fit(d, "y_dliv_1", CH + ["post", "fin_x_post", "real_x_post", "liv"], 1, "break",
                    "1986-1995 vs 2010-2016, implied vol"))
    return pd.DataFrame(rows)


def table_robust(p):
    """Q7. Is the financial channel just the leverage effect, the Depression, or a lag choice?"""
    rows = []
    p = p.copy()
    p["ret"] = p["mkt_ret"]
    p["ret_neg"] = p["mkt_ret"].clip(upper=0)
    p["ret_3"] = p["mkt_ret"].rolling(3).sum()
    base = sub(p, ["pre_vix"])
    rows.append(fit(base, "y_lrv_1", CH + HAR + ["ret", "ret_neg", "ret_3"], 1, "robust",
                    "1926-1985, + past returns (leverage effect)"))
    rows.append(fit(base, "y_lrv_12", CH + HAR + ["ret", "ret_neg", "ret_3"], 12, "robust",
                    "1926-1985, + past returns, 12m"))
    rows.append(fit(base.loc["1946-01":], "y_lrv_1", CH + HAR, 1, "robust", "1946-1985 (drops Depression and war)"))
    rows.append(fit(base.loc["1946-01":], "y_lrv_12", CH + HAR, 12, "robust", "1946-1985 (drops Depression and war), 12m"))
    rows.append(fit(base, "y_lrv_1", CH + HAR + ["nvix_z"], 1, "robust", "1926-1985, + total NVIX"))
    clean = sub(p, CLEAN_IV)
    rows.append(fit(clean, "y_lrv_1", CH + HAR + ["liv", "ret", "ret_neg", "ret_3"], 1, "robust",
                    "clean IV windows, + implied vol + past returns"))
    rows.append(fit(base, "y_lrv_1", CH + HAR, 1, "robust", "1926-1985, Newey-West 24 lags", lags=24))
    rows.append(fit(clean, "y_lrv_1", CH + HAR + ["liv"], 1, "robust", "clean IV windows, Newey-West 24 lags", lags=24))
    return pd.DataFrame(rows)


def oos_forecast(p, start_eval="1946-01", min_train=240):
    """Q6. Real-time forecasting of next-month log realised vol, expanding window, 1946 onward.
    Benchmark: HAR. Challenger: HAR + both news channels. Standardisation is recomputed
    each month from data available at the time, so the forecast uses no future information
    beyond what is baked into NVIX itself (see README on the text model's training window)."""
    y = "y_lrv_1"
    cols = HAR + CHANNELS
    d = p[[y] + cols + ["window"]].dropna()
    idx = d.index.tolist()
    out = []
    for i, m in enumerate(idx):
        if m < start_eval or i < min_train:
            continue
        tr = d.iloc[:i]
        # Only use rows whose target is fully observed by month m: target at row j uses month j+1.
        tr = tr.iloc[:-1]
        mu, sd = tr[CHANNELS].mean(), tr[CHANNELS].std()
        Xtr_b = sm.add_constant(tr[HAR])
        Xtr_c = sm.add_constant(pd.concat([tr[HAR], (tr[CHANNELS] - mu) / sd], axis=1))
        rb = sm.OLS(tr[y], Xtr_b).fit()
        rc = sm.OLS(tr[y], Xtr_c).fit()
        row = d.loc[[m]]
        fb = float(rb.predict(sm.add_constant(row[HAR], has_constant="add")).iloc[0])
        xc = pd.concat([row[HAR], (row[CHANNELS] - mu) / sd], axis=1)
        fc = float(rc.predict(sm.add_constant(xc, has_constant="add")).iloc[0])
        out.append({"month": m, "window": row["window"].iloc[0], "y": float(row[y].iloc[0]),
                    "f_har": fb, "f_news": fc})
    f = pd.DataFrame(out).set_index("month")
    f["e_har"] = (f["y"] - f["f_har"]) ** 2
    f["e_news"] = (f["y"] - f["f_news"]) ** 2
    # Clark and West (2007) adjusted MSPE difference for nested models.
    f["cw"] = f["e_har"] - (f["e_news"] - (f["f_har"] - f["f_news"]) ** 2)
    summ = []
    for name, g in [("1946-2016 all", f)] + list(f.groupby("window")):
        cw = g["cw"]
        ols = sm.OLS(cw, np.ones(len(cw))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        summ.append({"sample": name, "n": len(g),
                     "oos_r2_vs_har": 1 - g["e_news"].sum() / g["e_har"].sum(),
                     "clark_west_t": float(ols.tvalues.iloc[0]),
                     "clark_west_p_one_sided": float(norm.sf(ols.tvalues.iloc[0]))})
    f.to_csv(RES / "oos_forecasts.csv")
    return pd.DataFrame(summ)


def rolling_coef(p, window=180):
    """Rolling 15-year coefficient on each channel, realised-vol target, HAR controls."""
    d = p[["y_lrv_1"] + HAR + CHANNELS].dropna()
    rows = []
    for i in range(window, len(d) + 1):
        w = d.iloc[i - window:i]
        z = (w[CHANNELS] - w[CHANNELS].mean()) / w[CHANNELS].std()
        X = sm.add_constant(pd.concat([z, w[HAR]], axis=1))
        r = sm.OLS(w["y_lrv_1"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        rows.append({"end": d.index[i - 1], "b_fin": r.params["ch_financial"], "se_fin": r.bse["ch_financial"],
                     "b_real": r.params["ch_real"], "se_real": r.bse["ch_real"]})
    out = pd.DataFrame(rows).set_index("end")
    out.to_csv(RES / "rolling_coefficients.csv")
    return out


def main():
    p = load()
    tables = {
        "q1_quantity": table_quantity(p),
        "q2_implied": table_implied(p),
        "q3_price": table_price(p),
        "q4_categories": table_categories(p),
        "q5_break": table_break(p),
        "q7_robust": table_robust(p),
    }
    for k, t in tables.items():
        t.to_csv(RES / f"{k}.csv", index=False)
    oos = oos_forecast(p)
    oos.to_csv(RES / "q6_oos.csv", index=False)
    rolling_coef(p)
    reg = pd.DataFrame(REGISTRY)
    reg.to_csv(RES / "spec_registry.csv", index=False)
    pd.set_option("display.width", 250)
    for k, t in tables.items():
        keep = [c for c in t.columns if c in ("label", "y", "h", "n", "r2") or c.startswith(("b_ch", "t_ch", "b_cat", "t_cat", "pholm", "b_fin", "t_fin", "b_real", "t_real", "b_post", "t_post"))]
        print("\n==", k); print(t[keep].round(3).to_string(index=False))
    print("\n== q6_oos"); print(oos.round(4).to_string(index=False))
    print(f"\nspecifications estimated: {len(reg)}")


if __name__ == "__main__":
    main()
