"""Fetch the raw data into data/raw. Nothing is redistributed in this repository.

NVIX:   Manela and Moreira (2017), https://asaf.manela.org/data/ (free for non-commercial use)
VIX/VXO: CBOE via FRED (VIXCLS, VXOCLS)
Market:  Kenneth French data library, daily Fama-French factors (market excess return and T-bill)
"""
import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "nvix_and_categories_timeseries_mar2016.xlsx":
        "https://asaf.manela.org/papers/mm/nvix/nvix_and_categories_timeseries_mar2016.xlsx",
    "VIXCLS.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    "VXOCLS.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXOCLS",
}
FF_ZIP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nvix-volatility-channels research script"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        (RAW / name).write_bytes(get(url))
        print("fetched", name)
    with zipfile.ZipFile(io.BytesIO(get(FF_ZIP))) as z:
        z.extractall(RAW)
    print("fetched Fama-French daily factors")


if __name__ == "__main__":
    main()
