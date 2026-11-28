"""Step 7 - website data: schools, public transport and parcel coordinates as .js files.

The website is fully static (no server): open website/index.html or host the folder on GitHub Pages.
The data files are .js (they assign to a global variable) instead of .json, so no web server is needed to read them.
Coordinates are stored as whole numbers (degrees * 100,000, about 1 metre precision) to keep the files small.
"""
import json
import logging


from . import config as cfg
from .utils import load_json, read_csv

log = logging.getLogger(__name__)
COORD_SCALE = 100_000


def write_js(path, statement: str) -> None:
    path.write_text(statement, encoding="utf-8")
    log.info("wrote %s (%.2f MB)", path.name, path.stat().st_size / 1e6)


def write_poi() -> None:
    """poi.js:  window.RP_POI = {schools: [[lat, lon, name], ...], transit: [...]}"""
    poi = {}
    for layer in cfg.POI_LAYERS:
        table = read_csv(cfg.POI_CSV[layer["source"]]).fillna({"name": ""})
        poi[layer["source"]] = [[int(round(row.lat * COORD_SCALE)), int(round(row.lon * COORD_SCALE)), str(row.name)]
                                for row in table.itertuples()]
    write_js(cfg.WEBSITE_DATA_DIR / "poi.js",
             "window.RP_POI=" + json.dumps(poi, ensure_ascii=False, separators=(",", ":")) + ";")


def write_parcels() -> int:
    """parcels_<gush // PARCEL_CHUNK>.js:  {gush: [chelka, lat, lon, chelka, lat, lon, ...]}, sorted by chelka.

    The browser finds a chelka with a binary search (that is why it is sorted). The parcels are split into many
    small files that are loaded only when needed, so the first page load stays fast.
    Returns the number of parcels."""
    parcels = read_csv(cfg.PARCELS_CSV).sort_values(["gush", "chelka"])
    parcels["lat5"] = (parcels["lat"] * COORD_SCALE).round().astype(int)
    parcels["lon5"] = (parcels["long"] * COORD_SCALE).round().astype(int)

    for old_file in cfg.WEBSITE_DATA_DIR.glob("parcels_*.js"):   # remove files of gushim that no longer exist
        old_file.unlink()

    chunks = {}   # chunk number -> {gush: flat list}
    for gush, group in parcels.groupby("gush"):
        flat = group[["chelka", "lat5", "lon5"]].to_numpy().ravel().tolist()
        chunks.setdefault(int(gush) // cfg.PARCEL_CHUNK, {})[int(gush)] = flat

    for chunk_number, data in chunks.items():
        write_js(cfg.WEBSITE_DATA_DIR / f"parcels_{chunk_number}.js",
                 f"RP.parcelChunk({chunk_number},{json.dumps(data, separators=(',', ':'))});")
    return len(parcels)


def write_meta(n_parcels: int) -> None:
    """meta.js: settings and headline numbers that the page shows."""
    meta = {"layers": cfg.POI_LAYERS, "center": cfg.CITY_CENTER, "bbox": cfg.ISRAEL_BBOX,
            "parcelChunk": cfg.PARCEL_CHUNK, "nParcels": n_parcels, "nDeals": load_json(cfg.META_JSON)["n_rows"],
            "metrics": load_json(cfg.METRICS_JSON)["test"]}
    write_js(cfg.WEBSITE_DATA_DIR / "meta.js", "window.RP_META=" + json.dumps(meta, ensure_ascii=False) + ";")


def run() -> None:
    cfg.WEBSITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_poi()
    write_meta(write_parcels())
    log.info("website data ready in %s", cfg.WEBSITE_DATA_DIR)
