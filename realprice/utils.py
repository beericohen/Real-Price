"""Small helpers shared by several pipeline steps: logging, reading / writing files, column names."""
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd


def setup_logging() -> None:
    """Print INFO messages (with time) to the console. Called once by every entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")


# ---------------------------------------------------------------- reading and writing files
# Hebrew text: CSV files are written with "utf-8-sig" (UTF-8 + BOM) so Excel shows Hebrew correctly.
def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    """Read a csv / txt / xlsx file. Old Israeli files are often cp1255 (Windows Hebrew) instead of UTF-8."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def extract_zips(folder: Path) -> None:
    """Unpack every .zip in `folder` into the same folder (e.g. the 'download all as ZIP' archive)."""
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)


# ---------------------------------------------------------------- column names
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the column names, strip spaces / BOM and use "_" instead of spaces, so matching is forgiving."""
    df = df.copy()
    df.columns = [str(c).strip().strip("\ufeff").lower().replace(" ", "_") for c in df.columns]
    return df


def first_present(df: pd.DataFrame, candidates: list[str]):
    """The first name from `candidates` that is a column of `df` (or None)."""
    return next((c for c in candidates if c in df.columns), None)
