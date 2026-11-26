"""Loading of external geo data (data.gov.il): parcels (gush/chelka), schools and public transport.

Every source is resolved as: local file in data/external/<name>.*  ->  direct URL  ->  CKAN resource id.
Discover resources from the command line:   python -m realprice.external_data search "חלקות"
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as cfg
from .geo import inside_itm_bounds, itm_to_wgs84, looks_like_degrees, ring_centroid, wgs84_to_itm
from .utils import ensure_dirs, first_present, get_logger, normalise_columns, read_table

log = get_logger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (RealPrice research project)"}  # data.gov.il rejects empty agents
LOCAL_SUFFIXES = (".csv", ".txt", ".xlsx", ".xls", ".json", ".geojson", ".zip", ".shp", ".gpkg")


# ------------------------------------------------------------------ fetching
def _download(url: str, dest: Path) -> Path:
    log.info("downloading %s", url)
    ensure_dirs(dest.parent)
    with requests.get(url, stream=True, headers=HEADERS, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def _fetch_ckan_resource(resource_id: str, dest: Path, page: int = 32_000) -> Path:
    """Page through the CKAN datastore API and cache the whole table as CSV.
    If the resource has no datastore table, fall back to the file URL listed in its metadata."""
    frames, offset = [], 0
    while True:
        resp = requests.get(f"{cfg.CKAN_API}/datastore_search", headers=HEADERS, timeout=120,
                            params={"resource_id": resource_id, "limit": page, "offset": offset})
        if not resp.ok and offset == 0:
            meta = requests.get(f"{cfg.CKAN_API}/resource_show", headers=HEADERS, timeout=60,
                                params={"id": resource_id}).json()["result"]
            url = meta["url"]
            return _download(url, dest.with_suffix(Path(url.split("?")[0]).suffix or ".csv"))
        resp.raise_for_status()
        records = resp.json()["result"]["records"]
        if not records:
            break
        frames.append(pd.DataFrame(records))
        offset += page
    ensure_dirs(dest.parent)
    pd.concat(frames, ignore_index=True).to_csv(dest, index=False, encoding="utf-8-sig")
    return dest


def resolve_source(name: str) -> tuple[Path, str | None]:
    """Return (local file, zip member) for a configured source, downloading it when needed."""
    spec = cfg.EXTERNAL_SOURCES[name]
    for path in sorted(cfg.DATA_EXTERNAL.glob(f"{name}.*")):
        if path.suffix.lower() in LOCAL_SUFFIXES and not (cfg.REFRESH_EXTERNAL and (spec.get("url") or spec.get("ckan_resource_id"))):
            return path, spec.get("member")
    if spec.get("url"):
        suffix = Path(spec["url"].split("?")[0]).suffix or ".csv"
        return _download(spec["url"], cfg.DATA_EXTERNAL / f"{name}{suffix}"), spec.get("member")
    if spec.get("ckan_resource_id"):
        return _fetch_ckan_resource(spec["ckan_resource_id"], cfg.DATA_EXTERNAL / f"{name}.csv"), None
    raise FileNotFoundError(
        f"No data for '{name}'. Put a file named {name}.csv/.geojson/.xlsx in {cfg.DATA_EXTERNAL}, or set "
        f"EXTERNAL_SOURCES['{name}'] (url / ckan_resource_id) in realprice/config.py. "
        f'Search data.gov.il with: python -m realprice.external_data search "{name}"')


# ------------------------------------------------------------------ reading
def _read_geojson(path: Path) -> pd.DataFrame:
    """GeoJSON -> table of properties + polygon centroid in columns x / y (units of the file's CRS)."""
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    rows = []
    for feat in features:
        geom = feat.get("geometry") or {}
        coords, kind = geom.get("coordinates"), geom.get("type")
        if not coords:
            continue
        if kind == "Point":
            cx, cy = coords[:2]
        else:  # Polygon / MultiPolygon: centroid of the largest outer ring
            rings = [coords[0]] if kind == "Polygon" else [poly[0] for poly in coords]
            cx, cy, _ = max((ring_centroid(r) for r in rings), key=lambda c: c[2])
        rows.append({**(feat.get("properties") or {}), "x": cx, "y": cy})
    return pd.DataFrame(rows)


def read_external(path: Path, member: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = member or next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
            with zf.open(name) as fh:
                return pd.read_csv(fh, encoding="utf-8-sig", low_memory=False)
    if suffix in (".json", ".geojson"):
        return _read_geojson(path)
    if suffix in (".shp", ".gpkg"):
        try:
            import geopandas as gpd  # optional dependency, only for shapefiles
        except ImportError as exc:
            raise ImportError("Reading .shp/.gpkg needs geopandas (pip install geopandas) "
                              "- or convert the layer to GeoJSON/CSV.") from exc
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs(2039) if gdf.crs else gdf.set_crs(2039)
        cent = gdf.geometry.centroid
        return pd.DataFrame(gdf.drop(columns="geometry")).assign(x=cent.x.values, y=cent.y.values)
    return read_table(path)


# ------------------------------------------------------------------ schema detection
def extract_points(df: pd.DataFrame) -> pd.DataFrame:
    """Return x, y (ITM metres) and lat, lon (degrees) per row, index-aligned with `df` (NaN when invalid)."""
    alias = cfg.COLUMN_ALIASES
    lat_c, lon_c = first_present(df, alias["lat"]), first_present(df, alias["lon"])
    x_c, y_c = first_present(df, alias["x"]), first_present(df, alias["y"])
    if lat_c and lon_c:                       # WGS84 columns are the most explicit -> prefer them
        lat, lon = (pd.to_numeric(df[c], errors="coerce") for c in (lat_c, lon_c))
        x, y = wgs84_to_itm(lat, lon)
    elif x_c and y_c:
        a, b = (pd.to_numeric(df[c], errors="coerce") for c in (x_c, y_c))
        if looks_like_degrees(a, b):          # file in degrees: x = lon, y = lat
            lon, lat = a, b
            x, y = wgs84_to_itm(lat, lon)
        else:
            x, y = a.to_numpy(float), b.to_numpy(float)
    else:
        raise ValueError(f"Cannot find coordinate columns in {list(df.columns)}")
    out = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}, index=df.index)
    out.loc[~inside_itm_bounds(out["x"], out["y"]), ["x", "y"]] = np.nan
    out["lat"], out["lon"] = itm_to_wgs84(out["x"], out["y"])
    return out


def load_parcels() -> pd.DataFrame:
    """One row per (gush, chelka): ITM centroid in x / y."""
    df = normalise_columns(read_external(*resolve_source("parcels")))
    gush_c, chelka_c = first_present(df, cfg.COLUMN_ALIASES["gush"]), first_present(df, cfg.COLUMN_ALIASES["chelka"])
    if not (gush_c and chelka_c):
        raise ValueError(f"Parcel file needs gush/chelka columns, found {list(df.columns)}")
    pts = extract_points(df)
    parcels = pd.DataFrame({"gush": pd.to_numeric(df[gush_c], errors="coerce"),
                            "chelka": pd.to_numeric(df[chelka_c], errors="coerce"),
                            "x": pts["x"], "y": pts["y"]}).dropna()
    parcels[["gush", "chelka"]] = parcels[["gush", "chelka"]].astype("int64")
    # a parcel may consist of several polygons -> average their centroids
    return parcels.groupby(["gush", "chelka"], as_index=False)[["x", "y"]].mean()


def load_points(name: str) -> pd.DataFrame:
    """Schools / transit stops: columns x, y (ITM), lat, lon, name - invalid rows removed."""
    df = normalise_columns(read_external(*resolve_source(name)))
    pts = extract_points(df).dropna(subset=["x", "y"])
    name_c = first_present(df, cfg.COLUMN_ALIASES["name"])
    if name_c is None:    # unknown schema: take the first text column that looks like a name ("shem_mosad", "...name...", "שם...")
        name_c = next((c for c in df.columns if any(k in c for k in ("shem", "name", "שם"))), None)
    log.info("%s: name column = %s (columns: %s)", name, name_c, list(df.columns)[:25])
    pts["name"] = df.loc[pts.index, name_c].astype(str).str.strip() if name_c else ""
    return pts.drop_duplicates(["x", "y"]).reset_index(drop=True)


# ------------------------------------------------------------------ discovery helper (CLI)
def search_ckan(query: str, rows: int = 10) -> None:
    """Print datasets + resource ids matching `query` so they can be pasted into config.py."""
    resp = requests.get(f"{cfg.CKAN_API}/package_search", headers=HEADERS, timeout=60,
                        params={"q": query, "rows": rows})
    resp.raise_for_status()
    for pkg in resp.json()["result"]["results"]:
        print(f"\n# {pkg['title']}  ({pkg['name']})")
        for res in pkg["resources"]:
            print(f"  - {res['format']:8} id={res['id']}  {res.get('name', '')}\n    {res.get('url', '')}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        search_ckan(" ".join(sys.argv[2:]))
    else:
        print('usage: python -m realprice.external_data search "<query>"')
