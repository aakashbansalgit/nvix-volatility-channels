"""Build the monthly panel used by every analysis script.

Inputs (data/raw, fetched by src/download.py):
  nvix_and_categories_timeseries_mar2016.xlsx  Manela and Moreira (2017) NVIX and word categories
  VIXCLS.csv, VXOCLS.csv                       CBOE implied volatility, daily, from FRED
  F-F_Research_Data_Factors_daily.csv          Ken French daily market excess return and T-bill

Output: data/panel_monthly.csv, one row per month.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "panel_monthly.csv"

# Windows defined by how the NVIX text model was estimated (Manela and Moreira 2017):
# trained on 1996-2009 against VXO, tested on 1986-1995, applied to everything else.
WINDOWS = [
    ("pre_vix", "1926-07", "1985-12"),       # no implied vol exists; text model never saw it
    ("svr_test", "1986-01", "1995-12"),      # the paper's own hold-out window
    ("svr_train", "1996-01", "2009-12"),     # NVIX was fitted to VXO here: contaminated for VIX targets
    ("post_sample", "2010-01", "2016-03"),   # same model applied to new text after publication sample
]


def load_nvix() -> pd.DataFrame:
    df = pd.read_excel(RAW / "nvix_and_categories_timeseries_mar2016.xlsx", sheet_name="news_implied_volatility")
    df["month"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d").dt.to_period("M")
    df = df.rename(columns={
        "NVIX": "nvix",
        "Government": "cat_government",
        "Intermediation": "cat_intermediation",
        "Natural Disaster": "cat_disaster",
        "Securities Markets": "cat_securities",
        "War": "cat_war",
        "Unclassified": "cat_unclassified",
    }).drop(columns="Date")
    # The two channels from the original September 2026 test.
    df["ch_financial"] = df["cat_intermediation"] + df["cat_securities"]
    df["ch_real"] = df["cat_government"] + df["cat_war"] + df["cat_disaster"]
    return df.set_index("month")


def load_market_realized() -> pd.DataFrame:
    path = RAW / "F-F_Research_Data_Factors_daily.csv"
    lines = path.read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 5 or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts[:5])
    d = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    d["date"] = pd.to_datetime(d["date"], format="%Y%m%d")
    for c in ["mkt_rf", "rf"]:
        d[c] = d[c].astype(float)
    d["r"] = d["mkt_rf"] + d["rf"]  # total market return, percent per day
    d["month"] = d["date"].dt.to_period("M")
    g = d.groupby("month")
    out = pd.DataFrame({
        "n_days": g["r"].count(),
        # Annualised realised variance, in percent squared, comparable to VIX^2.
        "rv": g["r"].apply(lambda x: 252.0 * np.mean(x ** 2)),
        "mkt_ret": g["r"].apply(lambda x: 100.0 * (np.prod(1 + x / 100.0) - 1)),
    })
    out["rvol"] = np.sqrt(out["rv"])
    return out


def load_implied(name: str) -> pd.Series:
    d = pd.read_csv(RAW / f"{name}.csv", na_values=".")
    d["month"] = pd.to_datetime(d["observation_date"]).dt.to_period("M")
    s = d.dropna().groupby("month")[name].last()  # end-of-month level
    return s.rename(name.lower().replace("cls", ""))


def main() -> None:
    nvix = load_nvix()
    mkt = load_market_realized()
    vix = load_implied("VIXCLS")
    vxo = load_implied("VXOCLS")
    p = mkt.join(nvix, how="outer").join(vix, how="left").join(vxo, how="left")
    p = p.loc["1926-07":"2016-03"]
    # Implied vol: VXO is the series NVIX was trained on and starts in 1986; VIX from 1990.
    p["iv"] = p["vxo"].combine_first(p["vix"])
    p["lrvol"] = np.log(p["rvol"])
    p["liv"] = np.log(p["iv"])
    # Variance risk premium, Bollerslev, Tauchen and Zhou (2009) timing: implied minus past realised.
    p["vrp"] = p["iv"] ** 2 - p["rv"]
    p["window"] = None
    for name, a, b in WINDOWS:
        p.loc[a:b, "window"] = name
    p.index = p.index.astype(str)
    p.index.name = "month"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT)
    print(f"wrote {OUT} with {len(p)} months")
    print(p.groupby("window").agg(months=("rvol", "size"), nvix_missing=("nvix", lambda s: s.isna().sum()),
                                  iv_available=("iv", lambda s: s.notna().sum())))


if __name__ == "__main__":
    main()
