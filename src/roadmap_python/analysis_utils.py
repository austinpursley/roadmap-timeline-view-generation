from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
def project_root(start: Path | None = None) -> Path:
    """
    Find the project root by walking upward until pyproject.toml is found.

    Works great from notebooks/ and anywhere inside the repo.
    """
    p = (start or Path.cwd()).resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find pyproject.toml. Are you inside the project repo?")


ROOT = project_root()
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def data_path(*parts: str) -> Path:
    """Build a path under ./data."""
    return DATA.joinpath(*parts)


def figures_path(*parts: str) -> Path:
    """Build a path under ./figures (and ensure it exists)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES.joinpath(*parts)


# -----------------------------
# I/O helpers
# -----------------------------
def read_csv(*parts: str, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV from ./data/<parts...>.

    Example:
        df = read_csv("2_processed", "table.csv")
    """
    path = data_path(*parts)
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, *parts: str, index: bool = False, **kwargs: Any) -> Path:
    """
    Write a CSV to ./data/<parts...>, creating parent dirs.
    """
    path = data_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    return path


# -----------------------------
# Plot helpers
# -----------------------------
def set_mpl_defaults() -> None:
    """Opinionated-but-light defaults for notebooks and reports."""
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.grid": True,
            "axes.titlesize": "large",
            "axes.labelsize": "medium",
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, name: str, *, dpi: int = 300, tight: bool = True) -> Path:
    """
    Save a figure into ./figures.

    If `name` has no suffix, saves as PNG.
    """
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".png")
    out = figures_path(p.name)

    if tight:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(out, dpi=dpi)

    return out


# -----------------------------
# Logging / env info
# -----------------------------
def get_logger(name: str = "analysis") -> logging.Logger:
    """
    Simple console logger for notebooks and scripts.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def versions(
    packages: Iterable[str] = ("numpy", "pandas", "matplotlib", "scipy"),
) -> dict[str, str]:
    """
    Return a small version map you can drop into reports/notebooks.
    """
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out
