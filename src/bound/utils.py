__all__ = [
    "LOG_LEVEL",
    "TrainTimer",
    "build_grb_timeout_fld",
    "build_static_mixed_up",
    "construct_filename",
    "harmonic",
    "include_gurobi_timeout_field",
    "log_seeds",
    "set_random_seeds",
    "set_debug_mode",
]

import dill as pk
import io
import logging
from pathlib import Path
import random
import re
import sys
import time
from typing import NoReturn, Optional, Tuple

import numpy as np

import torch
from torch import Tensor

from .import _config as config
from . import dirs
from .datasets import tabular
from .types import TensorGroup

LOG_LEVEL = logging.DEBUG


# Intelligently select number of workers
gettrace = getattr(sys, 'gettrace', None)


def set_debug_mode(seed: int = 42) -> NoReturn:
    logging.warning("Debug mode enabled")
    config.enable_debug_mode()

    torch.manual_seed(seed)


def set_random_seeds() -> NoReturn:
    r"""
    Sets random seeds to avoid non-determinism
    :See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    seed = torch.initial_seed()
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False

    seed &= 2 ** 32 - 1  # Ensure a valid seed for the learner
    random.seed(seed)
    np.random.seed(seed)

    log_seeds()


def log_seeds():
    r""" Log the seed information """
    logging.debug("Torch Random Seed: %d", torch.initial_seed())
    # if "numpy" in sys.modules:
    #     state_str = re.sub(r"\s+", " ", str(np.random.get_state()))
    #     logging.debug("NumPy Random Seed: %s", state_str)
    #  Prints a seed way too long for normal use
    # if "random" in sys.modules:
    #     import random
    #     logging.debug("Random (package) Seed: %s", random.getstate())


def configure_dataset_args() -> TensorGroup:
    r""" Manages generating the source data (if not already serialized to disk """
    logging.debug("Reseed training set for consistent dataset creation given the seed")
    set_random_seeds()

    if config.DATASET.is_tabular():
        tabular_dir = dirs.DATA_DIR / config.DATASET.value.name.lower()
        tg = tabular.load_data(tabular_dir)
    else:
        raise ValueError(f"Dataset generation not supported for {config.DATASET.name}")
    x_shape = tg.tr_x.shape[1:]
    logging.info(f"Dataset Dimension: {np.prod(x_shape)}")
    return tg


def get_new_model(x: Optional[Tensor], opt_params: Optional[dict]):
    if config.DATASET.is_tabular():
        assert x is None, "X should not be specified for weather models"
        net = tabular.build_model()
    else:
        raise ValueError(f"Model creation not supported for {config.DATASET.name}")
    return net


def construct_filename(prefix: str, out_dir: Path, file_ext: str, model_num: Optional[int] = None,
                       add_timestamp: bool = False, add_ds_to_path: bool = True) -> Path:
    r""" Standardize naming scheme for the filename """
    fields = [
        prefix,
        config.DATASET.name.lower().replace("_", "-"),
    ]
    if config.IS_CLASSIFICATION:
        fields.append("class")
    # else:
    #     raise ValueError("Unknown learning environment")

    if model_num is not None:
        fields.append(f"m-id={model_num:04d}")
    if config.MIXUP_COPIES is not None and config.MIXUP_COPIES > 0:
        fields.append(f"mix={config.MIXUP_COPIES:03d}")
    if config.DEBUG:
        fields.append("dbg")

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".":
        # Add period before extension if not already specified
        file_ext = "." + file_ext
    fields[-1] += file_ext

    # Add the dataset name to better organize files
    if add_ds_to_path:
        out_dir /= config.DATASET.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)


def harmonic(n: int) -> float:
    r""" Calculates the harmonic number """
    val = 1.
    for i in range(2, n + 1):
        val += 1 / i
    return val


class TrainTimer:
    r""" Used for tracking the training time """
    def __init__(self, model_name: str, model_id: int):
        self._model_name = model_name
        self._model_id = model_id
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self._start_time
        flds = [self._model_name, f"Model # {self._model_id}",
                "Training Time:", f"{elapsed:.6f}"]
        logging.info(" ".join(flds))


class CpuUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def build_static_mixed_up(x: Tensor, y: Tensor,
                          include_orig: bool = False) -> Tuple[Tensor, Tensor]:
    r""" Construct static mixed up X and Y vectors """
    if config.MIXUP_COPIES is None or config.MIXUP_COPIES == 0:
        return x, y

    ret_np = False
    if isinstance(x, np.ndarray):
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        ret_np = True

    assert config.MIXUP_COPIES > 0, "MIXUP_COPIES cannot be negative"

    all_x, all_y = [], []
    if include_orig:
        all_x.append(x.clone())
        all_y.append(y.clone())
    for _ in range(config.MIXUP_COPIES):
        _x, _y = mixup_data(x=x, y=y)
        all_x.append(_x)
        all_y.append(_y)
    all_x, all_y = torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)

    if ret_np:
        all_x, all_y = all_x.numpy(), all_y.numpy()
    return all_x, all_y


def mixup_data(x, y, alpha=1.0):
    r""" Returns mixed inputs, pairs of targets, and lambda """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y


def build_grb_timeout_fld() -> str:
    r""" Timeout field to include about the gurobi timeout """
    return f"grb-to={config.GUROBI_TIMEOUT}"


def include_gurobi_timeout_field() -> bool:
    r""" Returns True if gurobi timeout should be in the filename field """
    flds = [
        config.SPREAD_DEGREE > 1,
        config.ALT_TYPE.is_multicover_model() and not config.IS_NO_MULTICOVER,
    ]
    return config.DEFAULT_GUROBI_TIMEOUT != config.GUROBI_TIMEOUT and any(flds)
