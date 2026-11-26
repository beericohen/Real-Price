"""Step 2 - enrichment: gush/chelka -> coordinates, distances to schools / public transport, derived columns.

Heavy work is done once per UNIQUE parcel (many deals share a parcel) and joined back, and distances use
KD-trees, so the whole step is roughly O((P + S + T) log n) instead of O(deals x POIs).
"""
import numpy as np
import pandas as pd

from . import config as cfg
from .external_data import load_parcels, load_points
from .geo import PointIndex, itm_to_wgs84, wgs84_to_itm
from .utils import get_logger, read_csv, timed, write_csv

log = get_logger(__name__)


def attach_coordinates(deals: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Left-join the parcel centroid (ITM metres); optionally fall back to the gush centroid."""
    merged = deals.merge(parcels, on=["gush", "chelka"], how="left")
    exact = merged["x"].notna()
    log.info("parcel match: %.1f%% of deals", 100 * exact.mean())
    if cfg.ALLOW_GUSH_FALLBACK:
        centroids = parcels.groupby("gush")[["x", "y"]].mean()
        for axis in ("x", "y"):
            merged[axis] = merged[axis].fillna(merged["gush"].map(centroids[axis]))
        log.info("after gush fallback: %.1f%% of deals located", 100 * merged["x"].notna().mean())
    return merged.dropna(subset=["x", "y"]).reset_index(drop=True)


def add_poi_features(locations: pd.DataFrame, layers: dict[str, PointIndex]) -> pd.DataFrame:
    """Nearest distance + count inside a radius for every POI layer (config.POI_LAYERS)."""
    for layer in cfg.POI_LAYERS:
        index = layers[layer["source"]]
        locations[layer["dist_col"]], _ = index.nearest(locations["x"], locations["y"])
        locations[layer["count_col"]] = index.count_within(locations["x"], locations["y"], layer["radius"])
    cx, cy = wgs84_to_itm(cfg.CITY_CENTER["lat"], cfg.CITY_CENTER["lon"])
    locations["distance_from_center"] = np.hypot(locations["x"] - cx, locations["y"] - cy)
    return locations


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """lat/long, age and split deal date."""
    df["lat"], df["long"] = itm_to_wgs84(df["x"], df["y"])
    df["age"] = (cfg.CURRENT_YEAR - df["year_built"]).clip(lower=0)
    df["deal_day"], df["deal_month"], df["deal_year"] = (df["deal_date"].dt.day, df["deal_date"].dt.month,
                                                         df["deal_date"].dt.year)
    return df


def _export_poi_tables(parcels: pd.DataFrame, pois: dict[str, pd.DataFrame]) -> None:
    """WGS84 tables consumed by the website build."""
    parcels = parcels.assign(lat=itm_to_wgs84(parcels["x"], parcels["y"])[0],
                             long=itm_to_wgs84(parcels["x"], parcels["y"])[1])
    write_csv(parcels[["gush", "chelka", "lat", "long"]], cfg.PARCELS_CSV)
    write_csv(pois["schools"][["lat", "lon", "name"]], cfg.SCHOOLS_CSV)
    write_csv(pois["transit"][["lat", "lon", "name"]], cfg.TRANSIT_CSV)


@timed
def run() -> pd.DataFrame:
    deals = read_csv(cfg.CLEANED_CSV, parse_dates=["deal_date"])
    parcels = load_parcels()
    pois = {layer["source"]: load_points(layer["source"]) for layer in cfg.POI_LAYERS}
    log.info("parcels: %d | %s", len(parcels), " | ".join(f"{k}: {len(v)}" for k, v in pois.items()))

    deals = attach_coordinates(deals, parcels)
    indexes = {name: PointIndex(p["x"], p["y"]) for name, p in pois.items()}
    locations = add_poi_features(deals[["gush", "chelka", "x", "y"]].drop_duplicates(["gush", "chelka"]), indexes)
    deals = deals.merge(locations.drop(columns=["x", "y"]), on=["gush", "chelka"], how="left")
    deals = add_derived_columns(deals)

    bbox = cfg.ISRAEL_BBOX
    deals = deals[deals["lat"].between(*bbox["lat"]) & deals["long"].between(*bbox["lon"])]
    dataset = deals[cfg.FINAL_COLUMNS + cfg.EXTRA_COLUMNS].reset_index(drop=True)
    write_csv(dataset, cfg.DATASET_CSV)
    _export_poi_tables(parcels, pois)
    log.info("final dataset: %d rows x %d columns -> %s", *dataset.shape, cfg.DATASET_CSV)
    return dataset
