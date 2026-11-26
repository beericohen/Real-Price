"""Small shared helpers: logging, timing, file IO."""
import json
import logging
import time
import zipfile
from functools import wraps
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def timed(func):
    """Log how long a pipeline step took."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        get_logger(func.__module__).info("finished %s in %.1fs", func.__name__, time.perf_counter() - start)
        return result
    return wrapper


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """lower-case, strip and replace spaces so column matching is forgiving."""
    df = df.copy()
    df.columns = [str(c).strip().strip("\ufeff").lower().replace(" ", "_") for c in df.columns]
    return df


def first_present(df: pd.DataFrame, candidates: list[str]):
    """Return the first candidate that is a column of df (or None)."""
    return next((c for c in candidates if c in df.columns), None)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read csv/tsv/txt/xlsx (Hebrew-safe encodings) into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(Path(path).parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj, path: Path) -> None:
    ensure_dirs(Path(path).parent)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_zips(folder: Path) -> None:
    """Unpack any .zip found in `folder` (e.g. the 'download all as ZIP' archive) next to itself."""
    for archive in Path(folder).glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
