__all__ = [
    "NEG_LABEL",
    "POS_LABEL",
    "TEST_X_KEY",
    "TEST_Y_KEY",
    "TR_X_KEY",
    "TR_Y_KEY",
    "ToFloatAndNormalize",
    "VAL_X_KEY",
    "VAL_Y_KEY",
    "download_from_google_drive",
    "get_paths",
    "in1d",
    "print_stats",
    "scale_expm1_y",
]

import logging
from pathlib import Path
import tarfile
from typing import NoReturn, Tuple

import gdown

import torch
from torch import BoolTensor, Tensor
import torch.nn as nn

from .. import _config as config
from ..types import TensorGroup


POS_LABEL = +1
NEG_LABEL = 0

TR_X_KEY = "tr_x"
TR_Y_KEY = "tr_y"
VAL_X_KEY = "val_x"
VAL_Y_KEY = "val_y"
TEST_X_KEY = "test_x"
TEST_Y_KEY = "test_y"


def get_paths(base_dir: Path, data_dir: Path) -> Tuple[Path, Path]:
    r""" Reduce the training and test sets based on a fixed divider of the ordering """
    # Location to store the pruned data
    prune_dir = base_dir / "pruned"
    prune_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    # div = int(round(1 / config.VALIDATION_SPLIT_RATIO))
    for is_train in [True, False]:
        # Load the complete source data
        base_fname = "training" if is_train else "test"
        # Support two different file extensions
        for file_ext in (".pth", ".pt"):
            path = data_dir / (base_fname + file_ext)
            if not path.exists():
                continue
            paths.append(path)
            break
        else:
            raise ValueError("Unable to find processed tensor")

    return paths[0], paths[1]


# def calc_hash_tensor(np_arr: np.ndarray) -> LongTensor:
#     r""" Hashes a numpy array along the first dimension of \p np_arr """
#     hash_vals = []
#     for i in range(np_arr.shape[0]):
#         str_val = str(np_arr[i].data.tobytes())
#         hash_obj = hashlib.sha256(str_val.encode())
#
#         digest = hash_obj.hexdigest()
#         # Torch longs are only 8 bytes so restrict digest to a maximum of 8 bytes
#         digest = digest[-min(8, len(digest)):]
#         # Convert to an integer.  Need to specify the number as base 16
#         digest_int = int(digest, 16)
#
#         hash_vals.append(digest_int)
#
#     hash_tensor = torch.tensor(hash_vals, dtype=torch.long).long()
#     return hash_tensor


def in1d(ar1, ar2) -> BoolTensor:
    r""" Returns \p True if each element in \p ar1 is in \p ar2 """
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def download_from_google_drive(dest: Path, gd_url: str, file_name: str,
                               decompress: bool = False) -> NoReturn:
    r"""
    Downloads the source data from Google Drive

    :param dest: Folder to which the dataset is downloaded
    :param gd_url: Google drive file url
    :param file_name: Filename to store the downloaded file
    :param decompress: If \p True (and \p file_name has extension ".tar.gz"), unzips the downloaded
                       zip file
    """
    full_path = dest / file_name
    if full_path.exists():
        logging.info(f"File \"{full_path}\" exists.  Skipping download")
        return

    # Define the output files
    dest.mkdir(exist_ok=True, parents=True)
    gdown.download(url=gd_url, output=str(full_path), quiet=config.QUIET)
    if file_name.endswith(".tar.gz"):
        if decompress:
            with tarfile.open(str(full_path), "r") as tar:
                tar.extractall(path=str(dest))
    else:
        assert not decompress, "Cannot decompress a non tar.gz file"


class ToFloatAndNormalize(nn.Module):
    def __init__(self, normalize_factor: float):
        super().__init__()
        self._factor = normalize_factor

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out.div_(self._factor)
        return out


def print_stats(tg: TensorGroup, n_bins: int = 20) -> NoReturn:
    r""" Prints a simple perturbation histogram to understand the extent of each perturbation """
    assert n_bins > 0, "Number of bins must be positive"

    # Find a consistent min and max range
    y_vals = torch.cat([tg.tr_y, tg.test_y], dim=0)
    if config.DATASET.is_expm1_scale():
        y_vals = scale_expm1_y(y=y_vals)
    min_y, max_y = y_vals.min().item(), y_vals.max().item()
    # Print statistics on the y values
    for ds_prefix, ds_name in [("tr", "Train"), ("test", "Test")]:
        y = tg.__getattribute__(f"{ds_prefix}_y")
        logging.info(f"{config.DATASET.name} {ds_name} Dataset Size: {y.numel()}")

        if config.DATASET.is_expm1_scale():
            y = scale_expm1_y(y=y)
        if config.IS_CLASSIFICATION:
            _log_prior(ds_name=ds_name, y=y)


def _log_prior(ds_name: str, y: Tensor) -> NoReturn:
    r""" Calculate the prior """
    assert config.IS_CLASSIFICATION, "Calculating prior but not running classification"
    n_pos, n_ele = (y == POS_LABEL).sum(), y.numel()
    logging.info(f"{ds_name} Positive Prior: {n_pos / n_ele:.2%}")


def scale_expm1_y(y: Tensor) -> Tensor:
    r""" Large disparity in price values so housing prices are scaled. """
    return torch.expm1(y)
