"""Step 3 - exploratory data analysis: every plot is a small function registered with @plot('file-name')."""
import matplotlib

matplotlib.use("Agg")  # headless: write PNG files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .utils import ensure_dirs, get_logger, read_csv, timed

log = get_logger(__name__)
PLOTS = []  # (file name, function) registry filled by the decorator
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
MAX_POINTS = 60_000  # scatter plots use a random sample to stay fast


def plot(name):
    def register(func):
        PLOTS.append((name, func))
        return func
    return register


def _binned_median(x: pd.Series, y: pd.Series, bins) -> pd.DataFrame:
    """Median / quartiles of y per bin of x."""
    grouped = y.groupby(pd.cut(x, bins), observed=True)
    out = grouped.quantile([.25, .5, .75]).unstack()
    out.index = [iv.mid for iv in out.index]
    return out


@plot("01_price_distribution")
def price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["deal_amount"] / 1e6, bins=80, color="#2457d6")
    axes[0].set(title="Deal amount", xlabel="million NIS")
    axes[1].hist(np.log(df["deal_amount"]), bins=80, color="#d6882a")
    axes[1].set(title="log(deal amount) - the model target", xlabel="log NIS")
    return fig


@plot("02_price_per_sqm_distribution")
def ppsm_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["price_per_sqrm"], bins=100, color="#2457d6")
    ax.set(title="Price per sqm", xlabel="NIS / sqm")
    return fig


@plot("03_area_vs_price")
def area_vs_price(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(df["asset_area"], df["deal_amount"], gridsize=60, xscale="log", yscale="log",
                   bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, label="deals (log)")
    ax.set(title="Area vs price (log-log)", xlabel="asset area (sqm)", ylabel="deal amount (NIS)")
    return fig


@plot("04_rooms_boxplot")
def rooms_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rooms = sorted(df["room_num"].round().unique())
    ax.boxplot([df.loc[df["room_num"].round() == r, "price_per_sqrm"] for r in rooms],
               tick_labels=[f"{r:g}" for r in rooms], showfliers=False)
    ax.set(title="Price per sqm by number of rooms", xlabel="rooms", ylabel="NIS / sqm")
    return fig


@plot("05_age_vs_price_per_sqm")
def age_vs_ppsm(df):
    stats = _binned_median(df["age"], df["price_per_sqrm"], np.arange(0, 125, 5))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stats.index, stats[.5], color="#2457d6", label="median")
    ax.fill_between(stats.index, stats[.25], stats[.75], alpha=.25, color="#2457d6", label="IQR")
    ax.set(title="Price per sqm by building age", xlabel="age (years)", ylabel="NIS / sqm")
    ax.legend()
    return fig


@plot("06_market_over_time")
def market_over_time(df):
    monthly = df.assign(period=df["deal_year"] * 100 + df["deal_month"]).groupby("period").agg(
        deals=("deal_amount", "size"), ppsm=("price_per_sqrm", "median"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(monthly)), monthly["deals"], color="#c9d4f2")
    ax.set(ylabel="deals per month", title="Market activity and median price per sqm")
    ax2 = ax.twinx()
    ax2.plot(range(len(monthly)), monthly["ppsm"], color="#d6882a")
    ax2.set_ylabel("median NIS / sqm")
    ticks = range(0, len(monthly), max(1, len(monthly) // 12))
    ax.set_xticks(list(ticks), [str(monthly.index[i]) for i in ticks], rotation=45)
    return fig


@plot("07_deal_nature")
def deal_nature(df):
    top = df["deal_nature"].value_counts().head(12).index
    stats = df[df["deal_nature"].isin(top)].groupby("deal_nature")["price_per_sqrm"].agg(["size", "median"])
    stats = stats.sort_values("size")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    labels = [str(x)[::-1] for x in stats.index]  # matplotlib has no bidi support -> reverse Hebrew for display
    axes[0].barh(labels, stats["size"], color="#2457d6")
    axes[0].set(title="Deals per type", xlabel="deals")
    axes[1].barh(labels, stats["median"], color="#d6882a")
    axes[1].set(title="Median price per sqm", xlabel="NIS / sqm")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    return fig


@plot("08_correlation_heatmap")
def correlation(df):
    cols = [c for c in df.select_dtypes("number").columns if c not in ("gush", "chelka", "sub_chelka")]
    corr = df[cols].assign(log_price=np.log(df["deal_amount"])).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    return fig


@plot("09_map_price_per_sqm")
def price_map(df):
    sample = df.sample(min(len(df), MAX_POINTS), random_state=cfg.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(6, 9))
    sc = ax.scatter(sample["long"], sample["lat"], c=np.log(sample["price_per_sqrm"]), s=3, cmap="plasma")
    fig.colorbar(sc, label="log(price per sqm)")
    ax.set(title="Where the deals are", xlabel="longitude", ylabel="latitude", aspect=1.2)
    return fig


@plot("10_distance_effects")
def distance_effects(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, layer in zip(axes, cfg.POI_LAYERS):
        stats = _binned_median(df[layer["dist_col"]], df["price_per_sqrm"], np.arange(0, 2050, 100))
        ax.plot(stats.index, stats[.5], color="#2457d6")
        ax.fill_between(stats.index, stats[.25], stats[.75], alpha=.25, color="#2457d6")
        ax.set(title=f"Price per sqm vs {layer['dist_col']}", xlabel="metres", ylabel="NIS / sqm")
    return fig


@plot("11_top_gush")
def top_gush(df):
    stats = df.groupby("gush")["price_per_sqrm"].agg(["size", "median"])
    stats = stats[stats["size"] >= 50].nlargest(20, "median").sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(stats.index.astype(str), stats["median"], color="#2457d6")
    ax.set(title="Most expensive gushim (>= 50 deals)", xlabel="median NIS / sqm", ylabel="gush")
    return fig


def _write_gallery(names: list[str]) -> None:
    """A tiny index.html so all charts can be browsed on one page."""
    imgs = "\n".join(f'<h3>{n}</h3><img src="{n}.png" style="max-width:100%">' for n in names)
    (cfg.EDA_DIR / "index.html").write_text(f"<meta charset='utf-8'><title>RealPrice EDA</title>{imgs}",
                                           encoding="utf-8")


@timed
def run() -> None:
    df = read_csv(cfg.DATASET_CSV)
    ensure_dirs(cfg.EDA_DIR)
    df.describe(include="all").T.to_csv(cfg.EDA_DIR / "summary_statistics.csv", encoding="utf-8-sig")
    for name, func in PLOTS:
        fig = func(df)
        fig.tight_layout()
        fig.savefig(cfg.EDA_DIR / f"{name}.png")
        plt.close(fig)
    _write_gallery([n for n, _ in PLOTS])
    log.info("saved %d charts to %s", len(PLOTS), cfg.EDA_DIR)
