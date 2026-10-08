"""Figures for the README. Run after analysis.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

WIN = [("1926-07", "1985-12", "#f2f2f2", "before VIX"),
       ("1986-01", "1995-12", "#dbe9f6", "text model hold-out"),
       ("1996-01", "2009-12", "#fde0c5", "text model trained"),
       ("2010-01", "2016-03", "#dbe9f6", "post-sample")]
FIN, REAL = "#1f5a99", "#b5532a"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def shade(ax):
    for a, b, c, _ in WIN:
        ax.axvspan(pd.Period(a).to_timestamp(), pd.Period(b).to_timestamp(), color=c, lw=0, zorder=0)


def fig_channels():
    p = pd.read_csv(ROOT / "data" / "panel_monthly.csv", index_col=0)
    p.index = pd.PeriodIndex(p.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    shade(ax)
    ax.plot(p.index, p["ch_financial"].rolling(6).mean(), color=FIN, lw=1.1, label="Financial news (securities + intermediation)")
    ax.plot(p.index, p["ch_real"].rolling(6).mean(), color=REAL, lw=1.1, label="Real-economy news (government + war + disasters)")
    ax.set_ylabel("Contribution to NVIX\n(6-month average)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for a, b, _, lab in WIN:
        mid = pd.Period(a).to_timestamp() + (pd.Period(b).to_timestamp() - pd.Period(a).to_timestamp()) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, lab, ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "channels.png", dpi=160)


def fig_horizons():
    q = pd.read_csv(RES / "q1_quantity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, lab in zip(axes, [q["label"].unique()[0], q["label"].unique()[1]]):
        g = q[q["label"] == lab]
        for v, c, n in [("ch_financial_z", FIN, "Financial"), ("ch_real_z", REAL, "Real economy")]:
            se = g[f"b_{v}"] / g[f"t_{v}"]
            ax.errorbar(g["h"] + (0.15 if c == REAL else -0.15), g[f"b_{v}"], yerr=1.96 * se, fmt="o-", color=c,
                        capsize=3, label=n)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("Effect of a 1 s.d. news shock\non future log realised vol")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "horizons.png", dpi=160)


def fig_rolling():
    r = pd.read_csv(RES / "rolling_coefficients.csv", index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    shade(ax)
    for b, s, c, n in [("b_fin", "se_fin", FIN, "Financial"), ("b_real", "se_real", REAL, "Real economy")]:
        ax.plot(r.index, r[b], color=c, lw=1.2, label=n)
        ax.fill_between(r.index, r[b] - 1.96 * r[s], r[b] + 1.96 * r[s], color=c, alpha=0.15, lw=0)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Rolling 15-year coefficient\n(next-month log realised vol)")
    ax.set_xlabel("End of 15-year window")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rolling.png", dpi=160)


def fig_oos():
    f = pd.read_csv(RES / "oos_forecasts.csv", index_col=0)
    f.index = pd.PeriodIndex(f.index, freq="M").to_timestamp()
    cum = (f["e_har"] - f["e_news"]).cumsum()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    shade(ax)
    ax.plot(cum.index, cum, color="#222", lw=1.2)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Cumulative squared-error\nreduction vs HAR")
    ax.set_xlabel("Real-time forecasts of next-month realised volatility, expanding window")
    fig.tight_layout()
    fig.savefig(FIG / "oos.png", dpi=160)


if __name__ == "__main__":
    fig_channels(); fig_horizons(); fig_rolling(); fig_oos()
    print("figures written to", FIG)
