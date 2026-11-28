"""Loads the external geographic data from data.gov.il: parcels (gush / chelka), schools and public transport.

Each source (see config.EXTERNAL_SOURCES) is looked up in this order:
    1. a local file  data/external/<name>.*   (also used as a cache of earlier downloads)
    2. a direct download URL
    3. a CKAN resource id (the data.gov.il API)

To find a resource id from the command line:   python -m realprice.external_data search "חלקות"
"""
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as cfg
from .geo import inside_itm_bounds, itm_to_wgs84, looks_like_degrees, ring_centroid, wgs84_to_itm
from .utils import first_present, normalise_columns, read_table, setup_logging

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (RealPrice research project)"}   # data.gov.il rejects requests without one
LOCAL_SUFFIXES = (".csv", ".txt", ".xlsx", ".xls", ".json", ".geojson", ".zip", ".shp", ".gpkg")


# ------------------------------------------------------------------ getting the file
def download_file(url: str, dest: Path) -> Path:
    log.info("downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=HEADERS, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 20):   # 1 MB at a time
                file.write(chunk)
    return dest


def fetch_ckan_resource(resource_id: str, dest: Path, page_size: int = 32_000) -> Path:
    """Download a whole table from the CKAN datastore API (page by page) and save it as CSV.
    If the resource has no datastore table, download the file that its metadata points to instead."""
    pages, offset = [], 0
    while True:
        response = requests.get(f"{cfg.CKAN_API}/datastore_search", headers=HEADERS, timeout=120,
                                params={"resource_id": resource_id, "limit": page_size, "offset": offset})
        if not response.ok and offset == 0:
            info = requests.get(f"{cfg.CKAN_API}/resource_show", headers=HEADERS, timeout=60,
                                params={"id": resource_id}).json()["result"]
            url = info["url"]
            return download_file(url, dest.with_suffix(Path(url.split("?")[0]).suffix or ".csv"))
        response.raise_for_status()
        records = response.json()["result"]["records"]
        if not records:
            break
        pages.append(pd.DataFrame(records))
        offset += page_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pages, ignore_index=True).to_csv(dest, index=False, encoding="utf-8-sig")
    return dest


def resolve_source(name: str):
    """Return (local file, name of the member inside a zip or None) for a configured source.
    Downloads the data first if it is not on disk yet."""
    spec = cfg.EXTERNAL_SOURCES[name]
    can_download = bool(spec.get("url") or spec.get("ckan_resource_id"))

    # 1. a local file (unless we were asked to refresh and we know where to download from)
    if not (cfg.REFRESH_EXTERNAL and can_download):
        for path in sorted(cfg.DATA_EXTERNAL.glob(f"{name}.*")):
            if path.suffix.lower() in LOCAL_SUFFIXES:
                return path, spec.get("member")

    # 2. a direct URL
    if spec.get("url"):
        suffix = Path(spec["url"].split("?")[0]).suffix or ".csv"
        return download_file(spec["url"], cfg.DATA_EXTERNAL / f"{name}{suffix}"), spec.get("member")

    # 3. the CKAN API
    if spec.get("ckan_resource_id"):
        return fetch_ckan_resource(spec["ckan_resource_id"], cfg.DATA_EXTERNAL / f"{name}.csv"), None

    raise FileNotFoundError(
        f"No data for '{name}'. Put a file named {name}.csv/.geojson/.xlsx in {cfg.DATA_EXTERNAL}, or set "
        f"EXTERNAL_SOURCES['{name}'] (url / ckan_resource_id) in realprice/config.py. "
        f'Search data.gov.il with: python -m realprice.external_data search "{name}"')


# ------------------------------------------------------------------ reading the file
def read_geojson(path: Path) -> pd.DataFrame:
    """GeoJSON -> a table with the properties of every feature plus its centre point in columns x / y."""
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    rows = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates, kind = geometry.get("coordinates"), geometry.get("type")
        if not coordinates:
            continue
        if kind == "Point":
            cx, cy = coordinates[:2]
        else:
            # Polygon or MultiPolygon: use the centre of the largest outer ring
            rings = [coordinates[0]] if kind == "Polygon" else [polygon[0] for polygon in coordinates]
            cx, cy, _ = max((ring_centroid(ring) for ring in rings), key=lambda c: c[2])
        rows.append({**(feature.get("properties") or {}), "x": cx, "y": cy})
    return pd.DataFrame(rows)


def read_shapefile(path: Path) -> pd.DataFrame:
    """.shp / .gpkg -> a table with the polygon centres in x / y (ITM). Needs the optional geopandas package."""
    try:
        import geopandas as gpd
    except ImportError as error:
        raise ImportError("Reading .shp/.gpkg needs geopandas (pip install geopandas) "
                          "- or convert the layer to GeoJSON/CSV.") from error
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(2039) if gdf.crs else gdf.set_crs(2039)
    centres = gdf.geometry.centroid
    return pd.DataFrame(gdf.drop(columns="geometry")).assign(x=centres.x.values, y=centres.y.values)


def read_external(path: Path, member: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = member or next(n for n in archive.namelist() if n.lower().endswith((".csv", ".txt")))
            with archive.open(name) as file:
                return pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
    if suffix in (".json", ".geojson"):
        return read_geojson(path)
    if suffix in (".shp", ".gpkg"):
        return read_shapefile(path)
    return read_table(path)


# ------------------------------------------------------------------ finding the coordinate columns
def extract_points(df: pd.DataFrame) -> pd.DataFrame:
    """Return x, y (ITM metres) and lat, lon (degrees) for every row, with the same index as `df`.
    Rows with impossible coordinates get NaN."""
    alias = cfg.COLUMN_ALIASES
    lat_col, lon_col = first_present(df, alias["lat"]), first_present(df, alias["lon"])
    x_col, y_col = first_present(df, alias["x"]), first_present(df, alias["y"])

    if lat_col and lon_col:
        # GPS columns are the clearest, so they are preferred
        lat, lon = (pd.to_numeric(df[c], errors="coerce") for c in (lat_col, lon_col))
        x, y = wgs84_to_itm(lat, lon)
    elif x_col and y_col:
        a, b = (pd.to_numeric(df[c], errors="coerce") for c in (x_col, y_col))
        if looks_like_degrees(a, b):   # the file holds degrees under the name x / y (x = longitude)
            x, y = wgs84_to_itm(lat=b, lon=a)
        else:
            x, y = a.to_numpy(float), b.to_numpy(float)
    else:
        raise ValueError(f"Cannot find coordinate columns in {list(df.columns)}")

    points = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}, index=df.index)
    points.loc[~inside_itm_bounds(points["x"], points["y"]), ["x", "y"]] = np.nan
    points["lat"], points["lon"] = itm_to_wgs84(points["x"], points["y"])
    return points


def load_parcels() -> pd.DataFrame:
    """One row per (gush, chelka) with the parcel's centre in ITM metres (columns x, y)."""
    df = normalise_columns(read_external(*resolve_source("parcels")))
    gush_col = first_present(df, cfg.COLUMN_ALIASES["gush"])
    chelka_col = first_present(df, cfg.COLUMN_ALIASES["chelka"])
    if not (gush_col and chelka_col):
        raise ValueError(f"Parcel file needs gush/chelka columns, found {list(df.columns)}")

    points = extract_points(df)
    parcels = pd.DataFrame({"gush": pd.to_numeric(df[gush_col], errors="coerce"),
                            "chelka": pd.to_numeric(df[chelka_col], errors="coerce"),
                            "x": points["x"], "y": points["y"]}).dropna()
    parcels[["gush", "chelka"]] = parcels[["gush", "chelka"]].astype("int64")
    # A parcel can consist of several polygons: use the average of their centres
    return parcels.groupby(["gush", "chelka"], as_index=False)[["x", "y"]].mean()


def load_points(name: str) -> pd.DataFrame:
    """Schools or transit stops: columns x, y (ITM), lat, lon and name. Rows without a valid location are removed."""
    df = normalise_columns(read_external(*resolve_source(name)))
    points = extract_points(df).dropna(subset=["x", "y"])

    name_col = first_present(df, cfg.COLUMN_ALIASES["name"])
    if name_col is None:
        # unknown layout: take the first column that looks like a name ("shem_mosad", "...name...", "שם...")
        name_col = next((c for c in df.columns if any(key in c for key in ("shem", "name", "שם"))), None)
    log.info("%s: name column = %s (columns: %s)", name, name_col, list(df.columns)[:25])

    points["name"] = df.loc[points.index, name_col].astype(str).str.strip() if name_col else ""
    return points.drop_duplicates(["x", "y"]).reset_index(drop=True)


# ------------------------------------------------------------------ command line helper
def search_ckan(query: str, rows: int = 10) -> None:
    """Print datasets and their resource ids, to copy into config.EXTERNAL_SOURCES."""
    response = requests.get(f"{cfg.CKAN_API}/package_search", headers=HEADERS, timeout=60,
                            params={"q": query, "rows": rows})
    response.raise_for_status()
    for package in response.json()["result"]["results"]:
        print(f"\n# {package['title']}  ({package['name']})")
        for resource in package["resources"]:
            print(f"  - {resource['format']:8} id={resource['id']}  {resource.get('name', '')}\n    {resource.get('url', '')}")


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        search_ckan(" ".join(sys.argv[2:]))
    else:
        print('usage: python -m realprice.external_data search "<query>"')
