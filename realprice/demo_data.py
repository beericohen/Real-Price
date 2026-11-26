"""Synthetic data generator for `python main.py --demo` (pipeline smoke test; NOT real market data)."""
import numpy as np
import pandas as pd

from . import config as cfg
from .geo import itm_to_wgs84, wgs84_to_itm
from .utils import ensure_dirs, get_logger

log = get_logger(__name__)
CITIES = [  # code, name, lat, lon, price level (NIS / sqm)
    (5000, "תל אביב -יפו", 32.0853, 34.7818, 62_000), (3000, "ירושלים", 31.7683, 35.2137, 38_000),
    (4000, "חיפה", 32.7940, 34.9896, 18_000), (7000, "לוד", 31.9516, 34.8958, 17_000),
    (8300, "ראשון לציון", 31.9730, 34.7925, 30_000), (9000, "באר שבע", 31.2530, 34.7915, 14_000),
]
NATURES = {"דירה בבית קומות": 1.0, "דירה בבית צמוד קרקע": 1.12, "דירת גג": 1.18, "דירת גן": 1.08, "דירה בבית משותף": 0.97}
GUSH_PER_CITY, PARCELS_PER_GUSH, DEALS_PER_CITY = 15, 60, 4_000


def _city_parcels(rng, code, lat, lon):
    """Gushim = clusters of parcels scattered around the city centre (ITM metres)."""
    cx, cy = wgs84_to_itm(lat, lon)
    rows = []
    for g in range(GUSH_PER_CITY):
        gx, gy = cx + rng.normal(0, 2000), cy + rng.normal(0, 2000)
        pts = np.column_stack([gx + rng.normal(0, 350, PARCELS_PER_GUSH), gy + rng.normal(0, 350, PARCELS_PER_GUSH)])
        rows += [{"gush": code // 10 * 100 + g + 1, "parcel": p + 1, "x": x, "y": y} for p, (x, y) in enumerate(pts)]
    return pd.DataFrame(rows)


def _deals(rng, parcels, code, name, level):
    pick = parcels.sample(DEALS_PER_CITY, replace=True, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    n = len(pick)
    area, rooms = rng.uniform(35, 190, n).round(), None
    rooms = np.clip(np.round(area / 28 + rng.normal(0, .5, n)), 1, 8)
    year = rng.integers(1950, cfg.CURRENT_YEAR, n)
    date = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 7, n), unit="D")
    nature = rng.choice(list(NATURES), n, p=[.55, .15, .1, .05, .15])
    ppsm = (level * np.exp(rng.normal(0, .12, n)) * (1 - .004 * np.clip(cfg.CURRENT_YEAR - year, 0, 60))
            * np.array([NATURES[x] for x in nature]) * 1.05 ** ((date.year + date.month / 12) - 2022)
            * (1 - .03 * np.abs(rooms - 4)))
    amount = (ppsm * area).round(-3)
    df = pd.DataFrame({"settlement_code": code, "settlement": name, "gush": pick["gush"], "chelka": pick["parcel"],
                       "sub_chelka": "000", "deal_date": date.strftime("%d/%m/%Y"), "deal_amount": amount,
                       "declared_amount": amount, "deal_nature": nature, "portion": 1.0, "year_built": year,
                       "asset_area": area, "room_num": rooms, "scraped_at": "2026-07-06 01:45:21+00"})
    # dirty rows so the cleaning step has something to do
    dirty = rng.choice(n, int(n * .05), replace=False)
    df.loc[dirty[:40], "room_num"] = np.nan
    df.loc[dirty[40:80], "year_built"] = 0
    df.loc[dirty[80:100], "deal_amount"] *= 25          # absurd price (outlier)
    df.loc[dirty[100:120], "asset_area"] = 0
    return pd.concat([df, df.head(20)], ignore_index=True)  # exact duplicates


def generate() -> None:
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    ensure_dirs(cfg.DATA_RAW, cfg.DATA_EXTERNAL)
    for old in list(cfg.DATA_RAW.glob("*")) + list(cfg.DATA_EXTERNAL.glob("*")):
        old.unlink()
    parcels = []
    for code, name, lat, lon, level in CITIES:
        city_parcels = _city_parcels(rng, code, lat, lon)
        parcels.append(city_parcels)
        _deals(rng, city_parcels, code, name, level).to_csv(cfg.DATA_RAW / f"deals_{code}.csv", index=False,
                                                           encoding="utf-8-sig")
    parcels = pd.concat(parcels, ignore_index=True)
    parcels.to_csv(cfg.DATA_EXTERNAL / "parcels.csv", index=False)

    def scatter(n, scale):  # random points around random parcels
        base = parcels.sample(n, replace=True, random_state=1)
        return base["x"].to_numpy() + rng.normal(0, scale, n), base["y"].to_numpy() + rng.normal(0, scale, n)
    sx, sy = scatter(250, 300)  # schools in ITM, with Hebrew-style headers (tests alias detection)
    pd.DataFrame({"X": sx, "Y": sy, "שם_מוסד": [f"בית ספר {i}" for i in range(250)]}).to_csv(
        cfg.DATA_EXTERNAL / "schools.csv", index=False, encoding="utf-8-sig")
    tx, ty = scatter(1200, 400)  # transit stops in WGS84, GTFS style headers
    lat, lon = itm_to_wgs84(tx, ty)
    pd.DataFrame({"stop_name": [f"תחנה {i}" for i in range(1200)], "stop_lat": lat, "stop_lon": lon}).to_csv(
        cfg.DATA_EXTERNAL / "transit.txt", index=False, encoding="utf-8-sig")
    log.info("demo data written to %s", cfg.WORKSPACE)
