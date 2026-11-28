"""Makes SYNTHETIC data for `python main.py --demo`. It lets you test the whole pipeline without the real files.
The numbers are invented, so the resulting model says nothing about the real market."""
import logging

import numpy as np
import pandas as pd

from . import config as cfg
from .geo import itm_to_wgs84, wgs84_to_itm

log = logging.getLogger(__name__)

# settlement code, name, latitude, longitude, typical price per sqm (NIS)
CITIES = [
    (5000, "תל אביב -יפו", 32.0853, 34.7818, 62_000), (3000, "ירושלים", 31.7683, 35.2137, 38_000),
    (4000, "חיפה", 32.7940, 34.9896, 18_000), (7000, "לוד", 31.9516, 34.8958, 17_000),
    (8300, "ראשון לציון", 31.9730, 34.7925, 30_000), (9000, "באר שבע", 31.2530, 34.7915, 14_000),
]
# deal type -> price multiplier
NATURES = {"דירה בבית קומות": 1.0, "דירה בבית צמוד קרקע": 1.12, "דירת גג": 1.18, "דירת גן": 1.08, "דירה בבית משותף": 0.97}
GUSH_PER_CITY, PARCELS_PER_GUSH, DEALS_PER_CITY = 15, 60, 4_000


def make_parcels(rng, code, lat, lon) -> pd.DataFrame:
    """Gushim = groups of parcels scattered around the city centre (positions in ITM metres)."""
    center_x, center_y = wgs84_to_itm(lat, lon)
    rows = []
    for g in range(GUSH_PER_CITY):
        gush_x, gush_y = center_x + rng.normal(0, 2000), center_y + rng.normal(0, 2000)
        xs = gush_x + rng.normal(0, 350, PARCELS_PER_GUSH)
        ys = gush_y + rng.normal(0, 350, PARCELS_PER_GUSH)
        gush_number = code // 10 * 100 + g + 1
        rows += [{"gush": gush_number, "parcel": p + 1, "x": x, "y": y} for p, (x, y) in enumerate(zip(xs, ys))]
    return pd.DataFrame(rows)


def make_deals(rng, parcels, code, name, price_level) -> pd.DataFrame:
    """Deals on random parcels. The price depends on area, rooms, building age, deal type and date."""
    picked = parcels.sample(DEALS_PER_CITY, replace=True, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    n = len(picked)
    area = rng.uniform(35, 190, n).round()
    rooms = np.clip(np.round(area / 28 + rng.normal(0, .5, n)), 1, 8)
    year = rng.integers(1950, cfg.CURRENT_YEAR, n)
    date = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 7, n), unit="D")
    nature = rng.choice(list(NATURES), n, p=[.55, .15, .1, .05, .15])

    price_per_sqm = (price_level * np.exp(rng.normal(0, .12, n))                    # random noise
                     * (1 - .004 * np.clip(cfg.CURRENT_YEAR - year, 0, 60))        # older = cheaper
                     * np.array([NATURES[x] for x in nature])                       # deal type
                     * 1.05 ** ((date.year + date.month / 12) - 2022)              # market grows 5% a year
                     * (1 - .03 * np.abs(rooms - 4)))                               # unusual room counts are cheaper
    amount = (price_per_sqm * area).round(-3)

    df = pd.DataFrame({"settlement_code": code, "settlement": name, "gush": picked["gush"], "chelka": picked["parcel"],
                       "sub_chelka": "000", "deal_date": date.strftime("%d/%m/%Y"), "deal_amount": amount,
                       "declared_amount": amount, "deal_nature": nature, "portion": 1.0, "year_built": year,
                       "asset_area": area, "room_num": rooms, "scraped_at": "2026-07-06 01:45:21+00"})

    # Add some dirty rows, so that the cleaning step has something to do
    dirty = rng.choice(n, int(n * .05), replace=False)
    df.loc[dirty[:40], "room_num"] = np.nan
    df.loc[dirty[40:80], "year_built"] = 0
    df.loc[dirty[80:100], "deal_amount"] *= 25     # absurd price
    df.loc[dirty[100:120], "asset_area"] = 0
    return pd.concat([df, df.head(20)], ignore_index=True)   # 20 exact duplicates


def generate() -> None:
    """Write fake deal files into data/raw and fake parcels / schools / transit stops into data/external."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    cfg.DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    for old_file in list(cfg.DATA_RAW.glob("*")) + list(cfg.DATA_EXTERNAL.glob("*")):
        old_file.unlink()

    all_parcels = []
    for code, name, lat, lon, price_level in CITIES:
        parcels = make_parcels(rng, code, lat, lon)
        all_parcels.append(parcels)
        deals = make_deals(rng, parcels, code, name, price_level)
        deals.to_csv(cfg.DATA_RAW / f"deals_{code}.csv", index=False, encoding="utf-8-sig")
    parcels = pd.concat(all_parcels, ignore_index=True)
    parcels.to_csv(cfg.DATA_EXTERNAL / "parcels.csv", index=False)

    def scatter_near_parcels(n, spread):
        """n random points close to random parcels (ITM metres)."""
        base = parcels.sample(n, replace=True, random_state=1)
        return base["x"].to_numpy() + rng.normal(0, spread, n), base["y"].to_numpy() + rng.normal(0, spread, n)

    # Schools in ITM with Hebrew-style headers (this also tests the column-name detection)
    x, y = scatter_near_parcels(250, 300)
    pd.DataFrame({"X": x, "Y": y, "שם_מוסד": [f"בית ספר {i}" for i in range(250)]}).to_csv(
        cfg.DATA_EXTERNAL / "schools.csv", index=False, encoding="utf-8-sig")

    # Transit stops in GPS coordinates with GTFS-style headers
    x, y = scatter_near_parcels(1200, 400)
    lat, lon = itm_to_wgs84(x, y)
    pd.DataFrame({"stop_name": [f"תחנה {i}" for i in range(1200)], "stop_lat": lat, "stop_lon": lon}).to_csv(
        cfg.DATA_EXTERNAL / "transit.txt", index=False, encoding="utf-8-sig")
    log.info("demo data written to %s", cfg.WORKSPACE)
