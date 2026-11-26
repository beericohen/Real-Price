"""Find out WHERE a gush gets lost:   python -m realprice.diagnose_gush 6400 [chelka]

Checks, in pipeline order: raw deal files -> parcel layer (data/external) -> parcels_wgs84.csv -> website chunk file.
A gush that is missing from the website but exists in an earlier stage tells you which step to fix.
"""
import re
import sys

from . import config as cfg
from .cleaning import load_raw_deals
from .external_data import load_parcels
from .utils import read_csv


def main(gush: int, chelka: int | None = None) -> None:
    raw = load_raw_deals()
    print(f"raw deal files : gush {gush} has {(raw['gush'] == gush).sum()} deals")
    parcels = load_parcels()
    sel = parcels[parcels["gush"] == gush]
    print(f"parcel layer   : {len(sel)} parcels with a valid location" + (f" | chelka {chelka}: {(sel['chelka'] == chelka).any()}" if chelka else ""))
    if sel.empty:
        near = sorted(parcels["gush"].unique(), key=lambda g: abs(g - gush))[:5]
        print(f"  -> the layer in data/external has no usable geometry for this gush (closest gushim present: {near}).")
        print("     Likely causes: the downloaded layer is partial / is a different layer, or coordinates were outside Israel's bounds.")
    chunk = gush // cfg.PARCEL_CHUNK
    path = cfg.WEBSITE_DATA_DIR / f"parcels_{chunk}.js"
    found = path.exists() and re.search(rf'[{{,]"{gush}":\[', path.read_text(encoding="utf-8")) is not None
    print(f"website chunk  : {path.name} exists={path.exists()} contains gush={found}")
    if cfg.PARCELS_CSV.exists():
        print(f"parcels_wgs84  : {(read_csv(cfg.PARCELS_CSV)['gush'] == gush).sum()} rows")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
