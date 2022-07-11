__all__ = [
    "calc",
]

import dataclasses
import dill as pk
import logging
from typing import NoReturn, Optional, Tuple, Union

import torch
from torch import LongTensor, Tensor

from . import utils
from .. import _config as config
from ..datasets import utils as ds_utils
from .. import dirs
from .. import learner_ensemble
from ..types import TensorGroup
from .. import utils as parent_utils

# Base pickling only supports pickle up to 4GB.  Use pickle protocol 4 for larger files. See:
# https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
PICKLE_PROTOCOL = 4


@dataclasses.dataclass
class RegressionResults:
    name: str
    x: Tensor
    y: Tensor
    ids: LongTensor
    full_yhat: Tensor = torch.zeros(0)
    yhat: Tensor = torch.zeros(0)

    def is_empty(self) -> bool:
        r""" Returns \p True if the dataset actually contains some results """
        return self.y.numel() == 0


def calc(model: learner_ensemble.EnsembleLearner, tg: TensorGroup) -> float:
    r"""
    Calculates and logs the results for the rotation results

    :param model: Ensemble learner
    :param tg: \p TensorGroup of the results
    """
    prefix = f"reg-res-info-{model.name().lower()}"
    path = parent_utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")

    if not path.exists():
        test_x, test_y, test_ids = _get_test_x_y(tg=tg)
        x_flds = (
            test_x,
        )

        base_msg = "Calculating regression results for the %s dataset"
        res_flds = (
            RegressionResults(name="Test", x=test_x, y=test_y, ids=test_ids),
        )
        if config.DATASET.is_expm1_scale():
            # Fix log1p renormalization
            for res in res_flds:
                res.y = ds_utils.scale_expm1_y(y=res.y)
        assert len(x_flds) == len(res_flds), "Mismatch in length of results fields"
        for x, ds_info in zip(x_flds, res_flds):
            # No validation set for housing.  If a dataset is empty, just skip it
            if ds_info.is_empty():
                continue

            msg = base_msg % ds_info.name
            logging.info(f"Starting: {msg}")

            # Calculate the prediction results
            with torch.no_grad():
                ds_info.full_yhat = model.forward_wide(x=x).cpu()
            if config.DATASET.is_expm1_scale():
                ds_info.full_yhat = ds_utils.scale_expm1_y(ds_info.full_yhat)
            # Get the final prediction
            ds_info.yhat = model.calc_prediction(ds_info.full_yhat)

            logging.info(f"COMPLETED: {msg}")

        # Dump the results for simpler analysis
        with open(str(path), "wb+") as f_out:
            # Specify the pickle protocol due to size limitations
            pk.dump(res_flds, f_out, protocol=PICKLE_PROTOCOL)

    with open(str(path), "rb") as f_in:
        ds_flds = pk.load(f_in)  # type: Tuple[RegressionResults]

    for bound_dist in config.BOUND_DIST:
        for ds_info in ds_flds:
            if ds_info.is_empty():
                continue
            cert_ratio = _log_ds_bounds(model=model, ds_info=ds_info, bound_dist=bound_dist,
                                        x=ds_info.x)
    # noinspection PyUnboundLocalVariable
    return cert_ratio


def _get_test_x_y(tg: TensorGroup) -> Tuple[Tensor, Tensor, LongTensor]:
    r""" Select the u.a.r. (without replacement) test set to consider. """
    keep_idx = utils.get_test_idx_to_analyze(tg=tg)
    return tg.test_x[keep_idx], tg.test_y[keep_idx], tg.test_ids[keep_idx]


def _log_ds_info_accuracy(model: learner_ensemble.EnsembleLearner,
                          ds_info: RegressionResults) -> NoReturn:
    r""" Long the accuracy information """
    base_header = f"{model.name()} {ds_info.name} %s Err."
    err_dist = ds_info.yhat - ds_info.y
    _base_log_all_results(base_header=base_header, y=ds_info.y, vals=err_dist,
                          bin_width=config.BIN_WIDTH, use_abs=True)


def _log_ds_bounds(model: learner_ensemble.EnsembleLearner, ds_info: RegressionResults,
                   x: Tensor, bound_dist: float) -> float:
    r""" Log the robustness bound """
    assert bound_dist > 0, "Bound distance is assumed positive"
    prefix = utils.build_bound_prefix(desc="reg", model=model, bound_dist=bound_dist,
                                      ds_name=ds_info.name, use_greedy_flag=config.USE_GREEDY)
    path = parent_utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")

    err = ds_info.yhat - ds_info.y
    bound_val = utils.calc_bound_val(dist=bound_dist, y=ds_info.y)

    coverage = None
    if model.is_multicover():
        ubound = ds_info.y + bound_val
        lbound = ds_info.y - bound_val
        coverage = utils.calc_coverage(desc="reg", ds_name=ds_info.name, bound_dist=bound_dist,
                                       lbound=lbound, ubound=ubound, model=model, x=x)

    # Bounds may take a while to compute
    if not path.exists():
        upper_bound_dist = (bound_val - err).clip(min=0)
        lower_bound_dist = (-bound_val - err).clip(max=0)

        bound_info = model.calc_bound(full_yhat=ds_info.full_yhat, ids=ds_info.ids,
                                      l_cutoff_dist=lower_bound_dist, u_cutoff_dist=upper_bound_dist,
                                      coverage=coverage)
        with open(path, "wb+") as f_out:
            # Specify the pickle protocol due to size limitations
            pk.dump(bound_info, f_out, protocol=PICKLE_PROTOCOL)

    with open(path, "rb") as f_in:
        bound_info = pk.load(f_in)
    # Handle the case where the gurobi results are not include
    bound_vals, grb_info = utils.extract_bound_results(bound_res=bound_info)

    bin_width = utils.get_bound_log_width()
    # Bound description log with or without the percent sign based on teh configuration.
    bound_str = utils.build_bound_str(bound_dist=bound_dist)
    base_header = f"{model.name()} {ds_info.name} %s Bound (d={bound_str})"
    _base_log_all_results(base_header=base_header, y=ds_info.y, vals=bound_vals,
                          bin_width=bin_width, use_abs=False)

    cert_ratio = utils.log_certification_ratio(model=model, err=err, y=ds_info.y,
                                               bound=bound_vals, bound_dist=bound_dist)

    return cert_ratio


def _base_log_all_results(base_header: str, y: Tensor, vals: Tensor,
                          bin_width: Union[float, int], use_abs: bool,
                          tot_count: Optional[int] = None) -> NoReturn:
    r""" Helper method for logging the results """
    base_header = base_header.replace("%)", "%%)")
    header = base_header % "All"
    if not _is_values_valid(header=header, vals=vals):
        return
    utils.print_results(header=header, vals=vals, bin_width=bin_width, use_abs=use_abs,
                        tot_count=tot_count)


def _is_values_valid(header: str, vals: Tensor) -> bool:
    r""" Returns \p False if the values \p Tensor is invalid for results analysis """
    if vals.numel() == 0:
        logging.info(f"{header}: No training elements. Skipping")
        return False
    if vals.numel() == 1:
        logging.info(f"{header}: Only a single element. Skipping")
        return False

    return True
