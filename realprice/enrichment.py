"""Step 2 - enrichment: add a location and surroundings to every deal.

For each deal we add:  coordinates (from gush / chelka),  distance to the nearest school and bus / train stop,
how many of each are close,  distance to central Tel Aviv,  the building's age and the split deal date.

Many deals share one parcel, so the distance work is done once per unique parcel and then joined back.
The nearest-point searches use KD-trees (see geo.py), which is far faster than comparing every deal to every school.
"""
import logging

import numpy as np
import pandas as pd

from . import config as cfg
from .external_data import load_parcels, load_points
from .geo import PointIndex, itm_to_wgs84, wgs84_to_itm
from .utils import read_csv, write_csv

log = logging.getLogger(__name__)


def attach_coordinates(deals: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Add the parcel centre (columns x, y in ITM metres) to every deal."""
    merged = deals.merge(parcels, on=["gush", "chelka"], how="left")
    log.info("parcel match: %.1f%% of deals", 100 * merged["x"].notna().mean())

    if cfg.ALLOW_GUSH_FALLBACK:
        # chelka not in the parcel layer -> use the average position of the whole gush
        gush_centres = parcels.groupby("gush")[["x", "y"]].mean()
        for axis in ("x", "y"):
            merged[axis] = merged[axis].fillna(merged["gush"].map(gush_centres[axis]))
        log.info("after gush fallback: %.1f%% of deals located", 100 * merged["x"].notna().mean())

    return merged.dropna(subset=["x", "y"]).reset_index(drop=True)


def add_poi_features(parcel_locations: pd.DataFrame, poi_indexes: dict) -> pd.DataFrame:
    """For every location: distance to the nearest point of each layer (school, transit), how many points
    are within the layer's radius, and the distance to the city centre."""
    x, y = parcel_locations["x"], parcel_locations["y"]
    for layer in cfg.POI_LAYERS:
        index = poi_indexes[layer["source"]]
        parcel_locations[layer["dist_col"]], _ = index.nearest(x, y)
        parcel_locations[layer["count_col"]] = index.count_within(x, y, layer["radius"])

    center_x, center_y = wgs84_to_itm(cfg.CITY_CENTER["lat"], cfg.CITY_CENTER["lon"])
    parcel_locations["distance_from_center"] = np.hypot(x - center_x, y - center_y)
    return parcel_locations


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """GPS coordinates, the building's age today, and the deal date split into day / month / year."""
    df["lat"], df["long"] = itm_to_wgs84(df["x"], df["y"])
    df["age"] = (cfg.CURRENT_YEAR - df["year_built"]).clip(lower=0)
    df["deal_day"] = df["deal_date"].dt.day
    df["deal_month"] = df["deal_date"].dt.month
    df["deal_year"] = df["deal_date"].dt.year
    return df


def save_website_tables(parcels: pd.DataFrame, pois: dict) -> None:
    """Parcels, schools and stops as GPS tables. The last pipeline step turns them into website files."""
    lat, long = itm_to_wgs84(parcels["x"], parcels["y"])
    write_csv(parcels.assign(lat=lat, long=long)[["gush", "chelka", "lat", "long"]], cfg.PARCELS_CSV)
    for source, points in pois.items():
        write_csv(points[["lat", "lon", "name"]], cfg.POI_CSV[source])


def run() -> pd.DataFrame:
    deals = read_csv(cfg.CLEANED_CSV, parse_dates=["deal_date"])
    parcels = load_parcels()
    pois = {layer["source"]: load_points(layer["source"]) for layer in cfg.POI_LAYERS}
    log.info("parcels: %d | %s", len(parcels), " | ".join(f"{name}: {len(points)}" for name, points in pois.items()))

    deals = attach_coordinates(deals, parcels)

    # Distances are computed once per unique parcel, then joined back to all of its deals.
    poi_indexes = {name: PointIndex(points["x"], points["y"]) for name, points in pois.items()}
    unique_parcels = deals[["gush", "chelka", "x", "y"]].drop_duplicates(["gush", "chelka"])
    parcel_features = add_poi_features(unique_parcels, poi_indexes)
    deals = deals.merge(parcel_features.drop(columns=["x", "y"]), on=["gush", "chelka"], how="left")
    deals = add_derived_columns(deals)

    # Drop deals whose coordinates fall outside Israel
    box = cfg.ISRAEL_BBOX
    deals = deals[deals["lat"].between(*box["lat"]) & deals["long"].between(*box["lon"])]

    dataset = deals[cfg.DATASET_COLUMNS].reset_index(drop=True)
    write_csv(dataset, cfg.DATASET_CSV)
    save_website_tables(parcels, pois)
    log.info("final dataset: %d rows x %d columns -> %s", *dataset.shape, cfg.DATASET_CSV)
    return dataset
