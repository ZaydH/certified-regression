__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "LOG_DIR",
    "MODELS_DIR",
    "PLOTS_DIR",
    "RES_DIR",
    "update_base_dir",
]

from pathlib import Path
from typing import NoReturn


BASE_DIR = None
DATA_DIR = None
LOG_DIR = None
MODELS_DIR = None
PLOTS_DIR = None
RES_DIR = None


def _update_all_paths():
    r""" Sets all path names based on the base directory """
    global BASE_DIR, DATA_DIR, LOG_DIR, MODELS_DIR, PLOTS_DIR, RES_DIR

    BASE_DIR = Path(".").absolute()

    DATA_DIR = BASE_DIR / ".data"
    LOG_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    PLOTS_DIR = BASE_DIR / "plots"
    RES_DIR = BASE_DIR / "res"


def update_base_dir() -> NoReturn:
    r""" Updates the base directory """
    _update_all_paths()


_update_all_paths()
