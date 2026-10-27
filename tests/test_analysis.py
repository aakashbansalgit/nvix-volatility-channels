"""Checks that need no downloaded data: target timing, the realised variance
calculation, and that the real-time forecast never uses a later month.

Run from the repo root: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis
import build_panel


def fake_panel(n=360, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("1950-01", periods=n, freq="M").astype(str)
    p = pd.DataFrame(index=months)
    for c in analysis.CHANNELS + analysis.CATEGORIES + ["nvix"]:
        p[c] = rng.normal(size=n)
    p["rv"] = np.exp(rng.normal(5, 0.5, size=n))
    p["liv"] = rng.normal(3, 0.3, size=n)
    p["vrp"] = rng.normal(size=n)
    p["window"] = "pre_vix"
    return p


def test_targets_look_forward_and_controls_look_back(tmp_path, monkeypatch):
    raw = fake_panel(48)
    raw.to_csv(tmp_path / "panel.csv")
    monkeypatch.setattr(analysis, "PANEL", tmp_path / "panel.csv")
    p = analysis.load()
    rv = raw["rv"].to_numpy()
    t = 20
    assert np.isclose(p["y_lrv_1"].iloc[t], np.log(rv[t + 1]) / 2)
    assert np.isclose(p["y_lrv_3"].iloc[t], np.log(rv[t + 1:t + 4].mean()) / 2)
    assert np.isclose(p["lrv_3"].iloc[t], np.log(rv[t - 2:t + 1].mean()) / 2)
    assert np.isnan(p["y_lrv_12"].iloc[-1])


def test_realised_variance_from_daily_returns(tmp_path, monkeypatch):
    # Two days in January 1990, in the layout of the French data library file.
    text = ("header line\n,Mkt-RF,SMB,HML,RF\n"
            "19900102,1.00,0,0,0.02\n"
            "19900103,-2.00,0,0,0.02\n\n")
    (tmp_path / "F-F_Research_Data_Factors_daily.csv").write_text(text, encoding="latin-1")
    monkeypatch.setattr(build_panel, "RAW", tmp_path)
    out = build_panel.load_market_realized()
    r = np.array([1.02, -1.98])
    assert np.isclose(out["rv"].iloc[0], 252 * np.mean(r ** 2))
    assert np.isclose(out["mkt_ret"].iloc[0], 100 * ((1 + r / 100).prod() - 1))


def test_newey_west_lags_cover_the_overlap():
    assert analysis.nw_lags(1) == 6
    assert analysis.nw_lags(12) == 24


def test_forecast_for_a_month_ignores_everything_after_it(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RES", tmp_path)
    raw = fake_panel()
    raw.to_csv(tmp_path / "panel.csv")
    monkeypatch.setattr(analysis, "PANEL", tmp_path / "panel.csv")
    p = analysis.load()
    m = p.index[300]

    analysis.oos_forecast(p, start_eval="1950-01", min_train=240)
    before = pd.read_csv(tmp_path / "oos_forecasts.csv", index_col=0)

    shocked = p.copy()
    later = shocked.index >= m
    shocked.loc[later, ["y_lrv_1"] + analysis.CHANNELS] *= 10
    shocked.loc[shocked.index > m, analysis.HAR] *= 10
    analysis.oos_forecast(shocked, start_eval="1950-01", min_train=240)
    after = pd.read_csv(tmp_path / "oos_forecasts.csv", index_col=0)

    earlier = before.index < m
    assert np.allclose(before.loc[earlier, ["f_har", "f_news"]], after.loc[earlier, ["f_har", "f_news"]])
    # the forecast made at m uses month m's own news, so it must move: the test can fail
    assert not np.isclose(before.loc[m, "f_news"], after.loc[m, "f_news"])
