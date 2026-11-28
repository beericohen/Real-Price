"""Finds out WHERE a gush gets lost:   python -m realprice.diagnose_gush 6400 [chelka]

It checks the places a gush passes through, in pipeline order:
raw deal files -> parcel layer (data/external) -> parcels_wgs84.csv -> website chunk file.
If a gush is missing from the website but present in an earlier place, that tells you which step to look at.
"""
import re
import sys

from . import config as cfg
from .cleaning import load_raw_deals
from .external_data import load_parcels
from .utils import read_csv, setup_logging


def main(gush: int, chelka: int | None = None) -> None:
    raw = load_raw_deals()
    print(f"raw deal files : gush {gush} has {(raw['gush'] == gush).sum()} deals")

    parcels = load_parcels()
    in_gush = parcels[parcels["gush"] == gush]
    message = f"parcel layer   : {len(in_gush)} parcels with a valid location"
    if chelka:
        message += f" | chelka {chelka}: {(in_gush['chelka'] == chelka).any()}"
    print(message)
    if in_gush.empty:
        closest = sorted(parcels["gush"].unique(), key=lambda g: abs(g - gush))[:5]
        print(f"  -> the layer in data/external has no usable geometry for this gush (closest gushim present: {closest}).")
        print("     Likely causes: the downloaded layer is partial / is a different layer, or coordinates were outside Israel's bounds.")

    chunk_file = cfg.WEBSITE_DATA_DIR / f"parcels_{gush // cfg.PARCEL_CHUNK}.js"
    found = chunk_file.exists() and re.search(rf'[{{,]"{gush}":\[', chunk_file.read_text(encoding="utf-8")) is not None
    print(f"website chunk  : {chunk_file.name} exists={chunk_file.exists()} contains gush={found}")

    if cfg.PARCELS_CSV.exists():
        print(f"parcels_wgs84  : {(read_csv(cfg.PARCELS_CSV)['gush'] == gush).sum()} rows")


if __name__ == "__main__":
    setup_logging()
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
