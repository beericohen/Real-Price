"""Step 7 - website data: schools, public transport and parcel coordinates as lazily loaded .js files.

The site is fully static ("serverless"): open website/index.html or host the folder on GitHub Pages / Netlify.
Data files are .js (assigning to window.*) instead of .json so no web server is required.
"""
import json

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import ensure_dirs, get_logger, load_json, read_csv, timed

log = get_logger(__name__)
E5 = 100_000  # coordinates are stored as integers = degrees * 1e5 (~1 m precision) to keep files small


def _dump_js(path, statement: str) -> None:
    path.write_text(statement, encoding="utf-8")
    log.info("wrote %s (%.2f MB)", path.name, path.stat().st_size / 1e6)


def _poi_points(csv_path) -> list:
    df = read_csv(csv_path).fillna({"name": ""})
    return [[int(round(r.lat * E5)), int(round(r.lon * E5)), str(r.name)] for r in df.itertuples()]


def write_poi() -> None:
    poi = {layer["source"]: _poi_points(getattr(cfg, f"{layer['source'].upper()}_CSV")) for layer in cfg.POI_LAYERS}
    _dump_js(cfg.WEBSITE_DATA_DIR / "poi.js",
             "window.RP_POI=" + json.dumps(poi, ensure_ascii=False, separators=(",", ":")) + ";")


def write_parcels() -> int:
    """parcels_<gush // CHUNK>.js : {gush: [chelka, lat, lon, chelka, lat, lon, ...]} sorted by chelka
    (the browser binary-searches it). Chunks are loaded on demand, so the first page load stays tiny."""
    parcels = read_csv(cfg.PARCELS_CSV).sort_values(["gush", "chelka"])
    parcels["lat5"], parcels["lon5"] = (parcels["lat"] * E5).round().astype(int), (parcels["long"] * E5).round().astype(int)
    for old in cfg.WEBSITE_DATA_DIR.glob("parcels_*.js"):
        old.unlink()
    chunks: dict[int, dict[int, list]] = {}
    for gush, group in parcels.groupby("gush"):
        flat = group[["chelka", "lat5", "lon5"]].to_numpy().ravel().tolist()
        chunks.setdefault(int(gush) // cfg.PARCEL_CHUNK, {})[int(gush)] = flat
    for key, data in chunks.items():
        _dump_js(cfg.WEBSITE_DATA_DIR / f"parcels_{key}.js",
                 f"RP.parcelChunk({key},{json.dumps(data, separators=(',', ':'))});")
    return len(parcels)


def write_meta(n_parcels: int) -> None:
    meta = {"layers": cfg.POI_LAYERS, "center": cfg.CITY_CENTER, "bbox": cfg.ISRAEL_BBOX,
            "parcelChunk": cfg.PARCEL_CHUNK, "nParcels": n_parcels, "nDeals": load_json(cfg.META_JSON)["n_rows"],
            "metrics": load_json(cfg.METRICS_JSON)["test"]}
    _dump_js(cfg.WEBSITE_DATA_DIR / "meta.js", "window.RP_META=" + json.dumps(meta, ensure_ascii=False) + ";")


@timed
def run() -> None:
    ensure_dirs(cfg.WEBSITE_DATA_DIR)
    write_poi()
    write_meta(write_parcels())
    log.info("website ready: open %s", cfg.WEBSITE_DIR / "index.html")
