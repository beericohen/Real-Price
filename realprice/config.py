"""Central configuration. Every tunable value of the pipeline lives here."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# WORKSPACE holds data/models/reports. `main.py --demo` points it to a throw-away folder.
WORKSPACE = Path(os.environ.get("REALPRICE_WORKSPACE", ROOT))
QUICK = os.environ.get("REALPRICE_QUICK") == "1"  # tiny training budget (demo / smoke tests)

# ---------------------------------------------------------------- folders
DATA_RAW = WORKSPACE / "data" / "raw"            # the 51 deal files (csv / xlsx / zip)
DATA_EXTERNAL = WORKSPACE / "data" / "external"  # parcels, schools, transit (cache)
DATA_PROCESSED = WORKSPACE / "data" / "processed"
MODELS_DIR = WORKSPACE / "models"
REPORTS_DIR = WORKSPACE / "reports"
EDA_DIR = REPORTS_DIR / "eda"
TRAINING_DIR = REPORTS_DIR / "training"
WEBSITE_DIR = ROOT / "website"                   # static site; data/*.js are generated
WEBSITE_DATA_DIR = WEBSITE_DIR / "data"

# ---------------------------------------------------------------- intermediate files
CLEANED_CSV = DATA_PROCESSED / "cleaned.csv"
DATASET_CSV = DATA_PROCESSED / "dataset.csv"      # the final, enriched dataset
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
META_JSON = DATA_PROCESSED / "meta.json"
FEATURES_FILE = DATA_PROCESSED / "features.joblib"   # train / test feature frames (leak-free)
BUILDER_FILE = DATA_PROCESSED / "feature_builder.joblib"  # fitted FeatureBuilder (train statistics)
PARCELS_CSV = DATA_PROCESSED / "parcels_wgs84.csv"
SCHOOLS_CSV = DATA_PROCESSED / "schools_wgs84.csv"
TRANSIT_CSV = DATA_PROCESSED / "transit_wgs84.csv"
MODEL_FILE = MODELS_DIR / "realprice_model.joblib"
METRICS_JSON = REPORTS_DIR / "metrics.json"

# ---------------------------------------------------------------- cleaning
CURRENT_YEAR = date.today().year
DROP_COLUMNS = ["settlement_code", "settlement", "scraped_at"]   # `portion` is KEPT: it is needed to keep only full-ownership deals
# Columns where 0 means "missing". (sub_chelka == 0 is a legitimate value ("000"), so it is excluded.)
NONZERO_COLUMNS = ["gush", "chelka", "deal_amount", "declared_amount", "asset_area", "room_num"]
# year_built is OPTIONAL: placeholder values (0 / 1900) become NaN instead of removing the deal, because on real listings
# (e.g. Yad2) the year is often unknown. The models are trained to work with it missing.
NUMERIC_COLUMNS = NONZERO_COLUMNS + ["year_built", "sub_chelka"]
OPTIONAL_COLUMNS = ["year_built"]
DATE_FORMAT = "%d/%m/%Y"
# Hard domain limits (inclusive) - obviously wrong rows are removed before the statistical filter.
VALID_RANGES = {
    "deal_amount": (200_000, 60_000_000),
    "asset_area": (15, 600),
    "room_num": (1, 12),
    "year_built": (1901, CURRENT_YEAR + 3),     # applied only to known years (NaN = unknown, kept)
    "price_per_sqrm": (3_000, 200_000),
}
# Only comparable "normal dwelling" deals are kept. An allow-list is safer than a block-list: the raw data holds ~45
# nature labels (land, offices, shops, warehouses, parking, farms, "combination" deals ...) that are not comparable.
RESIDENTIAL_NATURES = ["דירה בבית קומות", "ד. מגורים", "מגורים", "קוטג' דו משפחתי", "קוטג' חד משפחתי",
                       "קוטג' טורי", "בית בודד", "דירת גן", "דירת גג", "דירת נופש"]
FULL_OWNERSHIP_PORTION = 1.0   # keep only deals where the whole property was sold (deal_amount is then the full price)
NEW_BUILD_MAX_AGE = 1          # is_new_build = (deal_year - year_built) <= this

# ---------------------------------------------------------------- external data (data.gov.il)
# Resolution order for every source: local file data/external/<name>.* -> `url` -> CKAN `ckan_resource_id`.
# Find resource ids with:  python -m realprice.external_data search "חלקות"
REFRESH_EXTERNAL = False  # True = download every source again on each run (otherwise a cached copy in data/external is reused)
CKAN_API = "https://data.gov.il/api/3/action"
EXTERNAL_SOURCES = {
    "parcels": {"url": None, "ckan_resource_id": None, "member": None},
    "schools": {"url": None, "ckan_resource_id": "2055063b-93e8-4d6a-b2c5-c253d18c0cf6", "member": None},  # משרד החינוך - מוסדות חינוך
    "transit": {"url": "https://gtfs.mot.gov.il/gtfsfiles/israel-public-transportation.zip",
                "ckan_resource_id": None, "member": "stops.txt"},
}
# Candidate column names (after lower-casing / underscores) used to auto-detect the schema.
COLUMN_ALIASES = {
    "gush": ["gush", "gush_num", "gush_number", "gush_no", "גוש"],
    "chelka": ["chelka", "parcel", "parcel_num", "parcel_number", "parcel_no", "חלקה"],
    "lat": ["lat", "latitude", "stop_lat", "y_wgs84"],
    "lon": ["lon", "lng", "long", "longitude", "stop_lon", "x_wgs84"],
    "x": ["x", "x_itm", "easting", "x_coord", "coord_x", "centroid_x"],
    "y": ["y", "y_itm", "northing", "y_coord", "coord_y", "centroid_y"],
    "name": ["name", "stop_name", "school_name", "institution_name", "shem_mosad", "שם_מוסד", "שם", "שם_מוסד_חינוך"],  # lower-case: columns are lower-cased before matching
}
ALLOW_GUSH_FALLBACK = True  # deals whose chelka is missing in the parcel layer get the gush centroid

# Points-of-interest layers -> features. Shared by the pipeline AND the website (via meta.js).
POI_LAYERS = [
    {"source": "schools", "label": "מוסד לימוד", "icon": "🏫",
     "dist_col": "distance_from_school", "count_col": "schools_within_1km", "radius": 1000},
    {"source": "transit", "label": "תחבורה ציבורית", "icon": "🚌",
     "dist_col": "distance_from_public_transport", "count_col": "transit_within_500m", "radius": 500},
]
CITY_CENTER = {"lat": 32.0853, "lon": 34.7818}  # Tel Aviv - proxy for "distance from the metropolis"
ISRAEL_BBOX = {"lat": (29.4, 33.5), "lon": (34.2, 35.95)}

FINAL_COLUMNS = [
    "gush", "chelka", "sub_chelka", "lat", "long", "deal_amount", "year_built", "age", "asset_area",
    "price_per_sqrm", "room_num", "distance_from_school", "distance_from_public_transport",
    "deal_day", "deal_month", "deal_year", "deal_nature", "portion"
]
EXTRA_COLUMNS = ["schools_within_1km", "transit_within_500m", "distance_from_center"]

# ---------------------------------------------------------------- preprocessing / training
RANDOM_STATE = 42
TEST_SIZE = 0.15
MIN_NATURE_COUNT = 200   # rarer deal_nature values are merged into "other"
NATURE_PREFIX = "nature__"
TARGET_LOG = "log_price"
QUANTILES = {"low": 0.10, "high": 0.90}   # price range shown on the website (80% interval)

SEARCH_ITERATIONS = 3 if QUICK else 14    # random-search trials for gradient boosting
SEARCH_MAX_ROWS = 300_000
MAX_BOOST_ITER = 60 if QUICK else 800   # (web model search budget)
# Production hyper-parameters of the main models (see training.py)
HGB_PARAMS = dict(min_samples_leaf=20, l2_regularization=2.0, learning_rate=0.03, early_stopping=True,
                  max_iter=80 if QUICK else 1500, n_iter_no_change=40, validation_fraction=0.1)
CATBOOST_PARAMS = dict(iterations=80 if QUICK else 2500, learning_rate=0.06, depth=8, l2_leaf_reg=3.0,
                       loss_function="RMSE", od_type="Iter", od_wait=100, verbose=False, thread_count=-1)
LGBM_PARAMS = dict(n_estimators=80 if QUICK else 3000, learning_rate=0.03, num_leaves=127, min_child_samples=20,
                   reg_lambda=2.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
VAL_FRACTION = 0.15                       # slice of train used for early stopping + blend-weight fitting
BASELINE_MAX_ROWS = 60_000
PERMUTATION_ROWS = 4_000

# ---------------------------------------------------------------- split-aware feature engineering (features.py)
# Everything below is FITTED ON THE TRAIN SPLIT ONLY and then applied to train / test (no data leakage).
TARGET_PPSM = "log_ppsm"                  # model target: log(price per sqm); the price is exp(pred) * asset_area
OUTLIER_PCT = (0.01, 0.99)                # train-set percentiles used to cut price_per_sqrm / asset_area extremes
N_CLUSTERS = 50                           # KMeans spatial clusters on (lat, long)
TE_FOLDS = 5                              # out-of-fold target encoding folds (train rows never see their own target)
TE_SMOOTHING = 20                         # pseudo-count: groups with few deals are pulled to their parent's mean
COMP_NEIGHBOURS = (5, 15, 50)             # k nearest PAST deals
COMP_RADII_M = (250, 500, 1000)           # ... and past deals inside these radii
COMP_MAX_CANDIDATES = 300                 # nearest candidates fetched per deal before the time filter (radius stats are
                                          # computed over at most this many nearest past deals)
TREND_MONTHS = 3                          # area price trend = mean log(ppsm) of the TREND_MONTHS calendar months before the deal
TREND_MIN_DEALS = 5                       # fewer deals than this in the cluster -> fall back to the national trend
COMP_CHUNK = 20_000                       # rows per kNN batch (bounds memory)
YEAR_MASK_FRAC = 0.3                       # share of TRAIN rows whose year_built is hidden so the models also learn the "unknown year" case
REFIT_ON_ALL = True                       # after the honest test evaluation, rebuild features + refit on train+test

# ---------------------------------------------------------------- website export
PARCEL_CHUNK = 100   # gush numbers per parcel data file (loaded lazily by the browser)
