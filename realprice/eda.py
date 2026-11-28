"""Step 3 - exploratory data analysis (EDA): draws charts of the dataset into reports/eda/.

Every chart is one small function that takes the dataset and returns a matplotlib figure.
The list PLOTS at the bottom gives each function its file name.
"""
import logging

import matplotlib

matplotlib.use("Agg")   # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .utils import read_csv

log = logging.getLogger(__name__)
plt.rcParams.update(cfg.PLOT_STYLE)
MAX_POINTS = 60_000   # scatter plots draw a random sample, to stay fast
BLUE, ORANGE = "#2457d6", "#d6882a"


def binned_quartiles(x: pd.Series, y: pd.Series, bins) -> pd.DataFrame:
    """The 25th, 50th and 75th percentile of y inside each bin of x (index = the middle of the bin)."""
    quartiles = y.groupby(pd.cut(x, bins), observed=True).quantile([.25, .5, .75]).unstack()
    quartiles.index = [interval.mid for interval in quartiles.index]
    return quartiles


def price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["deal_amount"] / 1e6, bins=80, color=BLUE)
    axes[0].set(title="Deal amount", xlabel="million NIS")
    axes[1].hist(np.log(df["deal_amount"]), bins=80, color=ORANGE)
    axes[1].set(title="log(deal amount) - the model target", xlabel="log NIS")
    return fig


def ppsm_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["price_per_sqrm"], bins=100, color=BLUE)
    ax.set(title="Price per sqm", xlabel="NIS / sqm")
    return fig


def area_vs_price(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    hexes = ax.hexbin(df["asset_area"], df["deal_amount"], gridsize=60, xscale="log", yscale="log",
                      bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hexes, label="deals (log)")
    ax.set(title="Area vs price (log-log)", xlabel="asset area (sqm)", ylabel="deal amount (NIS)")
    return fig


def rooms_boxplot(df):
    rooms = sorted(df["room_num"].unique())   # the real values, including half rooms (3.5, 4.5 ...)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rooms)), 4.5))   # wider when there are many boxes
    ax.boxplot([df.loc[df["room_num"] == r, "price_per_sqrm"] for r in rooms],
               tick_labels=[f"{r:g}" for r in rooms], showfliers=False)
    ax.set(title="Price per sqm by number of rooms", xlabel="rooms", ylabel="NIS / sqm")
    return fig


def age_vs_ppsm(df):
    stats = binned_quartiles(df["age"], df["price_per_sqrm"], np.arange(0, 125, 5))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stats.index, stats[.5], color=BLUE, label="median")
    ax.fill_between(stats.index, stats[.25], stats[.75], alpha=.25, color=BLUE, label="IQR")
    ax.set(title="Price per sqm by building age", xlabel="age (years)", ylabel="NIS / sqm")
    ax.legend()
    return fig


def market_over_time(df):
    monthly = df.assign(period=df["deal_year"] * 100 + df["deal_month"]).groupby("period").agg(
        deals=("deal_amount", "size"), ppsm=("price_per_sqrm", "median"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(monthly)), monthly["deals"], color="#c9d4f2")
    ax.set(ylabel="deals per month", title="Market activity and median price per sqm")
    ax2 = ax.twinx()   # second y axis for the price line
    ax2.plot(range(len(monthly)), monthly["ppsm"], color=ORANGE)
    ax2.set_ylabel("median NIS / sqm")
    ticks = range(0, len(monthly), max(1, len(monthly) // 12))
    ax.set_xticks(list(ticks), [str(monthly.index[i]) for i in ticks], rotation=45)
    return fig


def deal_nature(df):
    top_types = df["deal_nature"].value_counts().head(12).index
    stats = df[df["deal_nature"].isin(top_types)].groupby("deal_nature")["price_per_sqrm"].agg(["size", "median"])
    stats = stats.sort_values("size")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    labels = [str(x)[::-1] for x in stats.index]   # matplotlib cannot draw right-to-left text, so reverse the Hebrew
    axes[0].barh(labels, stats["size"], color=BLUE)
    axes[0].set(title="Deals per type", xlabel="deals")
    axes[1].barh(labels, stats["median"], color=ORANGE)
    axes[1].set(title="Median price per sqm", xlabel="NIS / sqm")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    return fig


def correlation(df):
    columns = [c for c in df.select_dtypes("number").columns if c not in ("gush", "chelka", "sub_chelka")]
    corr = df[columns].assign(log_price=np.log(df["deal_amount"])).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(image)
    ax.set_title("Correlation matrix")
    return fig


def price_map(df):
    sample = df.sample(min(len(df), MAX_POINTS), random_state=cfg.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(6, 9))
    points = ax.scatter(sample["long"], sample["lat"], c=np.log(sample["price_per_sqrm"]), s=3, cmap="plasma")
    fig.colorbar(points, label="log(price per sqm)")
    ax.set(title="Where the deals are", xlabel="longitude", ylabel="latitude", aspect=1.2)
    return fig


def distance_effects(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, layer in zip(axes, cfg.POI_LAYERS):
        stats = binned_quartiles(df[layer["dist_col"]], df["price_per_sqrm"], np.arange(0, 2050, 100))
        ax.plot(stats.index, stats[.5], color=BLUE)
        ax.fill_between(stats.index, stats[.25], stats[.75], alpha=.25, color=BLUE)
        ax.set(title=f"Price per sqm vs {layer['dist_col']}", xlabel="metres", ylabel="NIS / sqm")
    return fig


def top_gush(df):
    stats = df.groupby("gush")["price_per_sqrm"].agg(["size", "median"])
    stats = stats[stats["size"] >= 50].nlargest(20, "median").sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(stats.index.astype(str), stats["median"], color=BLUE)
    ax.set(title="Most expensive gushim (>= 50 deals)", xlabel="median NIS / sqm", ylabel="gush")
    return fig


# (file name, function) - the charts are drawn in this order
PLOTS = [
    ("01_price_distribution", price_distribution),
    ("02_price_per_sqm_distribution", ppsm_distribution),
    ("03_area_vs_price", area_vs_price),
    ("04_rooms_boxplot", rooms_boxplot),
    ("05_age_vs_price_per_sqm", age_vs_ppsm),
    ("06_market_over_time", market_over_time),
    ("07_deal_nature", deal_nature),
    ("08_correlation_heatmap", correlation),
    ("09_map_price_per_sqm", price_map),
    ("10_distance_effects", distance_effects),
    ("11_top_gush", top_gush),
]


def write_gallery(names: list[str]) -> None:
    """A tiny index.html that shows all charts on one page."""
    images = "\n".join(f'<h3>{n}</h3><img src="{n}.png" style="max-width:100%">' for n in names)
    (cfg.EDA_DIR / "index.html").write_text(f"<meta charset='utf-8'><title>RealPrice EDA</title>{images}",
                                            encoding="utf-8")


def run() -> None:
    df = read_csv(cfg.DATASET_CSV)
    cfg.EDA_DIR.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(cfg.EDA_DIR / "summary_statistics.csv", encoding="utf-8-sig")
    for name, draw in PLOTS:
        fig = draw(df)
        fig.tight_layout()
        fig.savefig(cfg.EDA_DIR / f"{name}.png")
        plt.close(fig)
    write_gallery([name for name, _ in PLOTS])
    log.info("saved %d charts to %s", len(PLOTS), cfg.EDA_DIR)
