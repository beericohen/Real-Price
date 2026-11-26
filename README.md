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
Deal files (one ZIP per settlement, 51 files)
        |  clean_data         drop unused columns, empty / zero rows, outliers
        v
  clean deals
        |  geocode            gush / chelka / sub-chelka -> ITM -> lat / long (data.gov.il)
        |  add_features       distance to nearest school and public transport, counts nearby, age, price per m²
        v
  model table
        |  eda                graphs  ->  reports/eda/
        |  preprocess         one-hot encoding of deal nature, log(price per m²) target
        v
  numeric table
        |  train              gradient-boosted trees: median + 10th + 90th percentile models
        v
  trained models
        |  export_model       trees, POI and parcel lookups as JavaScript files
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
├── website/                   the website (static)
│   ├── index.html             the page (English)
│   ├── predict.html           redirects to index.html#valuation
│   ├── icon.png
│   ├── css/style.css          styling (dark / light theme)
│   ├── js/common.js           formatting, map helpers, parcel + geocoding lookups
│   ├── js/picker.js           gush / chelka / address picker
│   ├── js/predictor.js        feature vector + tree evaluation (runs the model in the browser)
│   ├── js/chart.js            SVG price-curve chart
│   ├── js/app.js              page logic (form, results, receipt, what-if, stats)
│   └── data/                  meta.js, poi.js, model.js, parcels_*.js (created by the export step)
├── src/                       the Python pipeline (one file per step + main.py)
├── data/                      raw deal files, cleaned deals, model table
├── models/                    trained models and their metrics
└── reports/eda/               EDA images + index.html
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
python src/main.py
```

This runs the whole pipeline and overwrites the graphs, the trained models and `website/data/`.

## Limitations

- Estimates are statistical, not professional appraisals. The model does not know the property's condition,
  floor, parking, view, renovations or the buyer / seller situation.
- Deals are reported transaction prices; the model has no information about the future of the market.
- Coverage is uneven: densely populated areas have far more deals than rural ones, so estimates there are
  less reliable. The "reliability" meter on the site is a heuristic (range width and location precision), not
  a statistical confidence interval.
- Gush / chelka lookup depends on the parcel data; if a chelka is missing the centre of the gush is used and the
  location is marked as approximate.
