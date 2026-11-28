"""All settings of the pipeline in one place.

Every other module reads its numbers and file paths from here, so a value
(a path, a threshold, a hyper-parameter) only ever has to be changed once.
"""
import os
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent

# `python main.py --demo` sets REALPRICE_DEMO=1 *before* this module is imported.
# Demo mode works on synthetic data in a throw-away folder, with tiny model sizes,
# and never touches the real data or the real website files.
DEMO = os.environ.get("REALPRICE_DEMO") == "1"
WORKSPACE = ROOT / "demo_workspace" if DEMO else ROOT

DATA_RAW = WORKSPACE / "data" / "raw"            # the 51 deal files (csv / xlsx / zip)
DATA_EXTERNAL = WORKSPACE / "data" / "external"  # parcels, schools, transit (downloaded once, then cached)
DATA_PROCESSED = WORKSPACE / "data" / "processed"
MODELS_DIR = WORKSPACE / "models"
EDA_DIR = WORKSPACE / "reports" / "eda"
TRAINING_DIR = WORKSPACE / "reports" / "training"
WEBSITE_DIR = ROOT / "website"
# The generated .js data files of the website. In demo mode they go to the workspace so that
# the demo can never overwrite the real model that the website ships with.
WEBSITE_DATA_DIR = WORKSPACE / "website_data" if DEMO else WEBSITE_DIR / "data"

# ---------------------------------------------------------------- files produced by the pipeline steps
CLEANED_CSV = DATA_PROCESSED / "cleaned.csv"              # step 1 output
DATASET_CSV = DATA_PROCESSED / "dataset.csv"              # step 2 output (cleaned + coordinates + distances)
FEATURES_FILE = DATA_PROCESSED / "features.joblib"        # step 4 output: train / test feature tables
BUILDER_FILE = DATA_PROCESSED / "feature_builder.joblib"  # step 4 output: the fitted FeatureBuilder
META_JSON = DATA_PROCESSED / "meta.json"                  # step 4 output: column lists, categories, sizes
PARCELS_CSV = DATA_PROCESSED / "parcels_wgs84.csv"        # gush / chelka -> lat / long, used by the website
SCHOOLS_CSV = DATA_PROCESSED / "schools_wgs84.csv"
TRANSIT_CSV = DATA_PROCESSED / "transit_wgs84.csv"
MODEL_FILE = MODELS_DIR / "realprice_model.joblib"        # step 5 output
METRICS_JSON = WORKSPACE / "reports" / "metrics.json"     # step 5 output

# ---------------------------------------------------------------- step 1: cleaning
CURRENT_YEAR = date.today().year
DATE_FORMAT = "%d/%m/%Y"
DROP_COLUMNS = ["settlement_code", "settlement", "scraped_at"]   # `portion` is kept: we need it to find full-ownership deals

# Columns where 0 means "missing". (sub_chelka == 0 is a real value, so it is not in this list.)
NONZERO_COLUMNS = ["gush", "chelka", "deal_amount", "declared_amount", "asset_area", "room_num"]
NUMERIC_COLUMNS = NONZERO_COLUMNS + ["year_built", "sub_chelka"]

# year_built is optional: placeholder years (0 / 1900) become "unknown" instead of deleting the deal,
# because users of the website often do not know the year. The models are trained to cope with it.
OPTIONAL_COLUMNS = ["year_built"]

# Hard limits (inclusive). Rows outside them are obviously wrong and are removed.
VALID_RANGES = {
    "deal_amount": (200_000, 60_000_000),
    "asset_area": (15, 600),
    "room_num": (1, 12),
    "year_built": (1901, CURRENT_YEAR + 3),
    "price_per_sqrm": (3_000, 200_000),
}

# Only normal dwellings are comparable. An allow-list is safer than a block-list: the raw data has about
# 45 deal types (land, offices, shops, warehouses, parking, farms ...) that we do not want.
RESIDENTIAL_NATURES = ["דירה בבית קומות", "ד. מגורים", "מגורים", "קוטג' דו משפחתי", "קוטג' חד משפחתי",
                       "קוטג' טורי", "בית בודד", "דירת גן", "דירת גג", "דירת נופש"]
FULL_OWNERSHIP_PORTION = 1.0   # keep only deals where the whole property was sold, so deal_amount is the full price
NEW_BUILD_MAX_AGE = 1          # a building is "new" if (deal_year - year_built) <= this

# ---------------------------------------------------------------- step 2: enrichment (external data from data.gov.il)
# Each source is looked up in this order: local file data/external/<name>.*  ->  `url`  ->  CKAN resource id.
# To find a CKAN resource id:  python -m realprice.external_data search "חלקות"
REFRESH_EXTERNAL = False   # True = download every source again instead of using the cached file
CKAN_API = "https://data.gov.il/api/3/action"
EXTERNAL_SOURCES = {
    "parcels": {"url": None, "ckan_resource_id": None, "member": None},
    "schools": {"url": None, "ckan_resource_id": "2055063b-93e8-4d6a-b2c5-c253d18c0cf6", "member": None},
    "transit": {"url": "https://gtfs.mot.gov.il/gtfsfiles/israel-public-transportation.zip",
                "ckan_resource_id": None, "member": "stops.txt"},   # `member` = which file to read inside the zip
}

# The files come in different layouts, so we look for these column names (lower-case, spaces -> "_").
COLUMN_ALIASES = {
    "gush": ["gush", "gush_num", "gush_number", "gush_no", "גוש"],
    "chelka": ["chelka", "parcel", "parcel_num", "parcel_number", "parcel_no", "חלקה"],
    "lat": ["lat", "latitude", "stop_lat", "y_wgs84"],
    "lon": ["lon", "lng", "long", "longitude", "stop_lon", "x_wgs84"],
    "x": ["x", "x_itm", "easting", "x_coord", "coord_x", "centroid_x"],
    "y": ["y", "y_itm", "northing", "y_coord", "coord_y", "centroid_y"],
    "name": ["name", "stop_name", "school_name", "institution_name", "shem_mosad", "שם_מוסד", "שם", "שם_מוסד_חינוך"],
}
ALLOW_GUSH_FALLBACK = True  # a deal whose chelka is not in the parcel layer gets the centre of its gush

# Points of interest. Each one becomes two model features (distance to the nearest + how many are close).
# The website reads this list too (through meta.js), so both sides always agree.
POI_LAYERS = [
    {"source": "schools", "label": "מוסד לימוד", "icon": "🏫",
     "dist_col": "distance_from_school", "count_col": "schools_within_1km", "radius": 1000},
    {"source": "transit", "label": "תחבורה ציבורית", "icon": "🚌",
     "dist_col": "distance_from_public_transport", "count_col": "transit_within_500m", "radius": 500},
]
POI_CSV = {"schools": SCHOOLS_CSV, "transit": TRANSIT_CSV}
CITY_CENTER = {"lat": 32.0853, "lon": 34.7818}   # Tel Aviv, used as "distance from the centre of the country"
ISRAEL_BBOX = {"lat": (29.4, 33.5), "lon": (34.2, 35.95)}   # deals outside this box are dropped

# Columns of dataset.csv, in this order.
DATASET_COLUMNS = [
    "gush", "chelka", "sub_chelka", "lat", "long", "deal_amount", "year_built", "age", "asset_area",
    "price_per_sqrm", "room_num", "distance_from_school", "distance_from_public_transport",
    "deal_day", "deal_month", "deal_year", "deal_nature", "portion",
    "schools_within_1km", "transit_within_500m", "distance_from_center",
]

# ---------------------------------------------------------------- step 4: preprocessing / features
RANDOM_STATE = 42
TEST_SIZE = 0.15
MIN_NATURE_COUNT = 200   # deal types with fewer deals than this are merged into "other"
NATURE_PREFIX = "nature__"
TARGET = "log_ppsm"      # what the models predict: log(price per sqm). Price = exp(prediction) * area.

# Everything below is learned from the TRAIN split only, then applied to train and test (no data leakage).
OUTLIER_PCT = (0.01, 0.99)   # cut price_per_sqrm / asset_area outside these train percentiles
N_CLUSTERS = 50              # KMeans clusters on (lat, long)
TE_FOLDS = 5                 # target encoding is "out-of-fold": a row never sees its own price
TE_SMOOTHING = 20            # groups with few deals are pulled towards their parent group's mean
COMP_NEIGHBOURS = (5, 15, 50)      # statistics of the k nearest earlier deals ...
COMP_RADII_M = (250, 500, 1000)    # ... and of earlier deals within these radii (metres)
COMP_MAX_CANDIDATES = 300    # how many nearest deals are fetched per deal before throwing away the later ones
COMP_CHUNK = 20_000          # deals processed per batch (limits memory use)
TREND_MONTHS = 3             # area trend = mean log price per sqm of this many months before the deal
TREND_MIN_DEALS = 5          # fewer deals than this in the cluster -> use the national trend instead
YEAR_MASK_FRAC = 0.3         # share of TRAIN rows whose year_built is hidden, so the model learns the "unknown year" case

# ---------------------------------------------------------------- step 5: training
# In demo mode the models are tiny so that the whole pipeline runs in about a minute.
QUANTILES = {"low": 0.10, "high": 0.90}   # the price range on the website is the 10th - 90th percentile
VAL_FRACTION = 0.15          # part of train used for early stopping and for the blend weights
REFIT_ON_ALL = True          # after the honest test evaluation, retrain on train + test for production
PERMUTATION_ROWS = 4_000     # rows used for the feature-importance plot
WEB_MAX_ITER = 60 if DEMO else 800   # number of trees of the website model

HGB_PARAMS = dict(min_samples_leaf=20, l2_regularization=2.0, learning_rate=0.03, early_stopping=True,
                  max_iter=80 if DEMO else 1500, n_iter_no_change=40, validation_fraction=0.1)
CATBOOST_PARAMS = dict(iterations=80 if DEMO else 2500, learning_rate=0.06, depth=8, l2_leaf_reg=3.0,
                       loss_function="RMSE", od_type="Iter", od_wait=100, verbose=False, thread_count=-1)
LGBM_PARAMS = dict(n_estimators=80 if DEMO else 3000, learning_rate=0.03, num_leaves=127, min_child_samples=20,
                   reg_lambda=2.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

# ---------------------------------------------------------------- charts
# One look for every chart of the project (EDA charts and training charts), applied with plt.rcParams.update().
PLOT_STYLE = {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
              "axes.spines.top": False, "axes.spines.right": False}

# ---------------------------------------------------------------- step 7: website data
PARCEL_CHUNK = 100   # gush numbers per parcels_*.js file (the browser loads them on demand)
