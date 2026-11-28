# Real-Price

**Estimate the market price of a property in Israel, entirely in your browser.**

Live site: https://beericohen.github.io/Real-Price/


You locate a property by **gush / chelka** (block / parcel) or by address, enter its area, rooms, year built and
type, and a gradient-boosted tree model trained on **~1.7 million real Israeli deals** returns an estimated price
and range. There is no server and no backend: the trained trees are exported to JavaScript and run on the
visitor's device.

## What the site shows

- **Estimated price** with a likely range (the model's 10th-90th percentile)
- **Location picker** - gush / chelka / sub-chelka, free-text address, or a click on the map
- **Surroundings** - distance to the nearest school and public-transport stop, and to central Tel Aviv
- **Deal score** - compare an asking price with the estimate (Buying / Selling modes)
- **Market position** - how the property's price per m² ranks against the same property at ~48 spots within 3 km,
  plus a price-vs-area and price-vs-year-built curve drawn by the model
- **The same property, elsewhere** - the model's estimate 1.5 km away in each of 8 directions
- **Why this price?** - a "receipt" starting from a typical Tel Aviv flat and swapping in your location, area,
  rooms, year built and type one at a time; the rows add up exactly to the prediction
- **What-if simulator** - change area, rooms or year built and re-run the model instantly
- **Model facts** - accuracy metrics, factor importance and the model structure

> The market-position, curve and "elsewhere" views are **model outputs**, not real deals - the site ships the
> trained model, not the deal table. They are labelled as such on the page.

## How it works

```
Deal files (one file per settlement, 51 files)           python main.py runs these steps in order
        |  cleaning        realprice/cleaning.py        drop unused columns, empty / zero rows, impossible values
        v
  cleaned.csv
        |  enrichment      realprice/enrichment.py      gush / chelka -> coordinates (data.gov.il), distance to the
        |                                               nearest school / public transport, counts nearby, age
        v
  dataset.csv
        |  eda             realprice/eda.py             charts  ->  reports/eda/
        |  preprocessing   realprice/preprocessing.py   train / test split FIRST, then features learned from train only
        v                  realprice/features.py        (clusters, price trends, comparable deals, target encoding)
  features.joblib
        |  training        realprice/training.py        ensemble (HistGradientBoosting + CatBoost + LightGBM) and the
        |                                               website model: median + 10th + 90th percentile trees
        v
  trained models
        |  exporting       realprice/exporting.py       trees -> arrays the browser can evaluate (checked against sklearn)
        |  build_website   realprice/build_website.py   schools, stops and parcel coordinates as JavaScript files
        v
  website/data/*.js   ->   read by the website (website/js/)
```

### Model results (held-out deals)

| Metric | Value |
|---|---|
| R² on price | 0.9022 |
| R² on log price | 0.9250 |
| MAE (average error) | ₪169,020 |
| RMSE | ₪290,501 |
| Median absolute % error | 9.2% |
| Within 10% of the real price | 53.6% |
| Within 20% of the real price | 82.0% |
| Coverage of the 10-90% range | 79.0% (target 80%) |

The final model has **800 trees** for the median and the same structure for each percentile model, trained on
**1,634,870 deals** (trained 2026-11-26).

### Design choices

- **Predict log(price per m²).** A ₪100k error matters far more on a ₪1M flat than on a ₪5M villa. Training on the
  log of the price per square metre makes the model care about percentage error, and multiplying by the area
  turns the prediction back into a price.
- **Quantile models for the range.** Two extra models predict the 10th and 90th percentile, so the range widens
  where the market is less predictable instead of using a fixed ± error.
- **Location as coordinates plus context.** Latitude / longitude, distance from central Tel Aviv, distance to the
  nearest school and public transport, and how many of each are nearby.
- **Time as a feature.** Deal year / month / day and a continuous time index let the model follow market trends.
- **Exact receipt.** The "Why this price?" rows come from re-running the model with one factor changed at a time,
  not from an approximation.

## Exploratory data analysis (EDA)

All graphs are in [`reports/eda/`](reports/eda/) (open [`reports/eda/index.html`](reports/eda/index.html) to see
them on one page).

| Graph | What it shows |
|---|---|
| `01_price_distribution.png` | Deal prices are right-skewed |
| `02_price_per_sqm_distribution.png` | Distribution of price per m² |
| `03_area_vs_price.png` | Larger properties cost more |
| `04_rooms_boxplot.png` | Price by number of rooms |
| `05_age_vs_price_per_sqm.png` | How building age affects price per m² |
| `06_market_over_time.png` | Market trend over time |
| `07_deal_nature.png` | Deals by property type |
| `08_correlation_heatmap.png` | Correlations between features |
| `09_map_price_per_sqm.png` | Price per m² on the map |
| `10_distance_effects.png` | Effect of distance to schools, transport and the centre |
| `11_top_gush.png` | The most expensive / most active gush areas |

## Project structure

```
Real-Price/
├── main.py                    runs the pipeline (see "Retrain the model")
├── requirements.txt
├── realprice/                 the Python pipeline, one module per step
│   ├── config.py              every path, threshold and hyper-parameter in one place
│   ├── utils.py               small helpers (logging, reading / writing files)
│   ├── cleaning.py            step 1
│   ├── external_data.py       loads parcels, schools and transit stops from data.gov.il
│   ├── geo.py                 ITM <-> GPS conversion and the nearest-point search (KD-tree)
│   ├── enrichment.py          step 2
│   ├── eda.py                 step 3
│   ├── features.py            leak-free feature engineering (the FeatureBuilder class)
│   ├── preprocessing.py       step 4
│   ├── training.py            step 5
│   ├── exporting.py           step 6
│   ├── build_website.py       step 7
│   ├── demo_data.py           synthetic data for `python main.py --demo`
│   └── diagnose_gush.py       finds where a missing gush gets lost
├── website/                   the website (static, Hebrew, right-to-left)
│   ├── index.html             the page
│   ├── predict.html           redirects to index.html#valuation (old address)
│   ├── icon.png
│   ├── css/style.css          styling (dark / light theme)
│   ├── js/common.js           formatting, distances, spatial index, parcel + address lookups, map
│   ├── js/picker.js           gush / chelka / address picker
│   ├── js/predictor.js        feature vector + tree evaluation (runs the model in the browser)
│   ├── js/chart.js            SVG price-curve chart
│   ├── js/analysis.js         deal score, reliability, receipt, price curves (calculations only)
│   ├── js/app.js              the page itself: form, results, what-if, statistics (DOM only)
│   └── data/                  meta.js, poi.js, model.js, parcels_*.js (created by steps 6 and 7)
├── data/                      raw deal files and the intermediate tables (not in git)
├── models/                    trained models (not in git)
└── reports/                   EDA charts, training charts and metrics.json
```

The shipped data: **1,698,771** deals, **1,095,973** mapped parcels, **22,705** schools and **34,307**
public-transport stops (about 31 MB of JavaScript data files; parcels are split into chunks and loaded on demand).

## Run it locally

**The website** - any static server works (opening `index.html` directly will not load the data files):

```bash
cd website
python -m http.server 8000
# open http://localhost:8000
```

The page loads Leaflet from unpkg, map tiles from ArcGIS and address search from OpenStreetMap Nominatim, so it
needs an internet connection for the map and address lookup. Gush / chelka lookup and the estimate itself work
from the local data files.

**Retrain the model:**

```bash
pip install -r requirements.txt
python main.py                          # the whole pipeline
python main.py --from-step training     # continue from a step (earlier results are reused)
python main.py --skip-eda               # without the charts
```

This overwrites the graphs, the trained models and `website/data/`. The raw deal files go into `data/raw/`
(`.csv`, `.xlsx` or `.zip`).

**Try the pipeline without the real data:**

```bash
python main.py --demo
```

generates a small synthetic data set in `demo_workspace/` and runs every step on it in about a minute, with tiny
models. It never touches the real data or `website/data/`, so it is safe to run at any time.

## Limitations

- Estimates are statistical, not professional appraisals. The model does not know the property's condition,
  floor, parking, view, renovations or the buyer / seller situation.
- Deals are reported transaction prices; the model has no information about the future of the market.
- Coverage is uneven: densely populated areas have far more deals than rural ones, so estimates there are
  less reliable. The "reliability" meter on the site is a heuristic (range width and location precision), not
  a statistical confidence interval.
- Gush / chelka lookup depends on the parcel data; if a chelka is missing the centre of the gush is used and the
  location is marked as approximate.
