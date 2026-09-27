"""Which dataset the loop, oracle and tasks work on: real IMD data (default) or the synthetic set (tests)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SANDBOX_PATH = "data/imd_tmax.csv"   # where the data appears inside every sandbox


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    description: str
    memory_mb: int   # default sandbox memory cap for this dataset

    @property
    def available(self) -> bool:
        return self.path.exists()


DATASETS = {
    "real": Dataset("real", ROOT / "data" / "imd_tmax_real.csv",
                    "REAL IMD Pune 1-degree gridded daily Tmax, 2015-2024 (~355 land grid points)", 1024),
    "synthetic": Dataset("synthetic", ROOT / "data" / "imd_tmax.csv",
                         "synthetic IMD-like daily Tmax, 2015-2024 (35 grid points)", 512),
}
_active = "real" if DATASETS["real"].available else "synthetic"


def use(name: str) -> Dataset:
    global _active
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    if not DATASETS[name].available:
        hint = "python data/prepare_real_data.py --download" if name == "real" else "python data/generate_data.py"
        raise FileNotFoundError(f"{DATASETS[name].path.name} not found - run: {hint}")
    _active = name
    return DATASETS[name]


def active() -> Dataset:
    return DATASETS[_active]


def read(path: Path | str | None = None) -> pd.DataFrame:
    """The whole dataset (cached per file version). Treat the result as read-only."""
    p = Path(path or active().path)
    return _read_cached(str(p), p.stat().st_mtime)


@lru_cache(maxsize=2)
def _read_cached(path: str, _mtime: float) -> pd.DataFrame:
    # CSV via pandas' own C parser: parquet needs pyarrow's dataset module, whose _csv DLL is
    # blocked by Windows Smart App Control on this machine (every read triggered a block popup).
    return pd.read_csv(path, parse_dates=["date"])
