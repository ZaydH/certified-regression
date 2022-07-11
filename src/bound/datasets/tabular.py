__all__ = [
    "load_data",
]

import dill as pk
import logging
from pathlib import Path
from typing import NoReturn

import torch
from torch import Tensor

from .types import SplitDataset
from . import utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils


GD_FILE_IDS = {
    SplitDataset.AMES_HOUSING: "1Nx9YLK3UtrOkfJ4sIv7UjNznIXna0JlX",
    SplitDataset.AUSTIN_HOUSING: "1IIdu7JO1ERzjBU7SCNkOORCgIJnhP1wm",
    SplitDataset.DIAMONDS: "1cQdRQz3kvpEBAdMI3A76Uw6DqgQPLZAe",
    SplitDataset.LIFE: "1FlnCvPufiEohOwNe9R29VAxqbelzU323",
    SplitDataset.SPAMBASE: "1xa-1o9TSFv1PZu_iOPeMqP3s2zS3s7bP",
    SplitDataset.WEATHER: "1P_yYfc7GnLvV8nV_lQCeN9nHDVyvRjZ_",
}

WEATHER_DIV = 10


def _filter_shifts_weather(tg: TensorGroup) -> NoReturn:
    r""" Filter the train set u.a.r. """
    n_tr = tg.tr_x.shape[0]

    if config.DATASET.is_weather():
        div = WEATHER_DIV
    else:
        raise ValueError("Unknown weather small divider")

    tr_idx = torch.randperm(n_tr)[:int(n_tr / div)]
    for tensor_name in ["x", "y", "ids", "lbls", "hash"]:
        field_name = f"tr_{tensor_name}"
        tensor = tg.__getattribute__(field_name)  # type: Tensor
        tensor = tensor[tr_idx]
        tg.__setattr__(field_name, tensor)


def load_data(data_dir: Path) -> TensorGroup:
    r""" Load the tabular data """
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=data_dir,
                                                  file_ext="pk", add_ds_to_path=False)
    logging.info(f"Tabular Data Path: {tg_pkl_path}")
    if not tg_pkl_path.exists():
        base_name = config.DATASET.value.name.lower().replace("_", "-")
        gd_name = base_name + "-raw.tar.gz"
        gd_path = data_dir / gd_name

        file_id = GD_FILE_IDS[config.DATASET]
        file_url = "https://drive.google.com/uc?id={}".format(file_id)
        # Preprocessed tensors stored on Google Drive
        utils.download_from_google_drive(dest=data_dir, gd_url=file_url,
                                         file_name=str(gd_path), decompress=True)
        assert gd_path.exists(), f"Downloaded dataset not found at \"{gd_path}\""

        # Standardized location of the decompressed tensors location
        tensor_path = data_dir / "tensors" / (base_name + ".pt")
        assert tensor_path.exists(), f"Decompressed tensors file not found \"{tensor_path}\""
        data_dict = torch.load(tensor_path)  # type: dict

        tg = TensorGroup()
        # Extract train -- Formed by combining train and dev sets
        tr_x_lst, tr_y_lst = [data_dict[utils.TR_X_KEY]], [data_dict[utils.TR_Y_KEY]]
        if data_dict[utils.VAL_X_KEY] is not None:
            tr_x_lst.append(data_dict[utils.VAL_X_KEY])
            tr_y_lst.append(data_dict[utils.VAL_Y_KEY])
        tg.tr_x, tg.tr_y = torch.cat(tr_x_lst, dim=0), torch.cat(tr_y_lst, dim=0)

        # Test set is used directly
        tg.test_x, tg.test_y = data_dict[utils.TEST_X_KEY], data_dict[utils.TEST_Y_KEY]

        # Copy the labels
        tg.tr_lbls, tg.test_lbls = tg.tr_y, tg.test_y

        tg.calc_tr_hash()
        tg.build_ids()
        # if config.DATASET.is_austin_housing():
        #     _filter_austin_housing(tg=tg)
        if config.DATASET.is_weather():
            _filter_shifts_weather(tg=tg)

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)

    if config.ALT_TYPE.is_knn():
        x_combo = torch.cat([tg.tr_x, tg.test_x], dim=0)
        min_x, _ = x_combo.min(dim=0)
        max_x, _ = x_combo.max(dim=0)
        eps = 1E-8
        diff = max_x - min_x + eps
        # min_x = min_x.unsqueeze(dim=0)
        tg.tr_x = (tg.tr_x - min_x) / diff
        tg.test_x = (tg.test_x - min_x) / diff

    utils.print_stats(tg=tg)

    return tg
