__all__ = [
    "calc",
]

import dataclasses
import dill as pk
import io
import logging
import sys
from typing import NoReturn, Tuple

import pycm

import torch
from torch import BoolTensor, LongTensor, Tensor

from . import utils
from .. import _config as config
from ..datasets import utils as ds_utils
from .. import dirs
from .. import learner_ensemble
from ..types import BoundReturnType, TensorGroup
from .. import utils as parent_utils

TE_CLN_DS = "test"
TE_ADV_DS = "test-adv"
TE_ADV_ONLY = "test-only-bd"


@dataclasses.dataclass
class ClassificationResults:
    name: str
    x: Tensor
    y: LongTensor
    lbls: LongTensor
    ids: LongTensor

    full_yhat: Tensor = torch.zeros(0)
    yhat: Tensor = torch.zeros(0)

    y_pred: LongTensor = torch.zeros(0)


@dataclasses.dataclass
class BoundInfo:
    true_lbl: int
    mask: BoolTensor
    dist: Tensor
    bound_res: BoundReturnType


def calc(model: learner_ensemble.EnsembleLearner, tg: TensorGroup) -> float:
    r"""
    Calculates and writes to disk the model's results when performing classification.

    :param model: Ensemble learner
    :param tg: \p TensorGroup of the results
    """
    # val_mask = learner_ensemble.get_validation_mask(hash_vals=tg.tr_hash,
    #                                                 n_ds_parts=model.n_models)
    # tr_mask = ~val_mask

    prefix = f"res-info-{model.name().lower()}"
    path = parent_utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")

    if not path.exists():
        base_msg = "Calculating classification results for the %s dataset"
        test_x, test_y, test_lbls, test_ids = _get_test_x_y(tg=tg)
        ds_flds = (
            # ClassificationResults(name="Train", x=tg.tr_x[tr_mask], y=tg.tr_y[tr_mask],
            #                       lbls=tg.tr_lbls[tr_mask], ids=tg.tr_ids[tr_mask]),
            # ClassificationResults(name="Valid", x=tg.tr_x[val_mask], y=tg.tr_y[val_mask],
            #                       lbls=tg.tr_lbls[val_mask], ids=tg.tr_ids[val_mask]),
            ClassificationResults(name="Test", x=test_x, y=test_y, lbls=test_lbls, ids=test_ids),
        )
        for ds_info in ds_flds:
            msg = base_msg % ds_info.name
            logging.info(f"Starting: {msg}")

            with torch.no_grad():
                ds_info.full_yhat = model.forward_wide(x=ds_info.x).cpu()
            # Get the final prediction
            ds_info.yhat = model.calc_prediction(ds_info.full_yhat)

            ds_info.y_pred = torch.full_like(ds_info.y, ds_utils.POS_LABEL)
            threshold = model.get_class_threshold()
            ds_info.y_pred[ds_info.yhat < threshold] = ds_utils.NEG_LABEL

            logging.info(f"COMPLETED: {msg}")

        # Dump the results for simpler analysis
        with open(str(path), "wb+") as f_out:
            pk.dump(ds_flds, f_out)

    with open(str(path), "rb") as f_in:
        ds_flds = pk.load(f_in)

    # Logs the confusion matrix
    for ds_info in ds_flds:
        _log_ds_info_accuracy(model=model, ds_info=ds_info)

    # Logs the robustness bounds
    for ds_info in ds_flds:
        vals = _log_ds_bounds(model=model, ds_info=ds_info)
    # noinspection PyUnboundLocalVariable
    return vals


def _get_test_x_y(tg: TensorGroup) -> Tuple[Tensor, LongTensor, LongTensor, LongTensor]:
    r""" Select the u.a.r. (without replacement) test set to consider. """
    keep_idx = utils.get_test_idx_to_analyze(tg=tg)
    return tg.test_x[keep_idx], tg.test_y[keep_idx], tg.test_lbls[keep_idx], tg.test_ids[keep_idx]


def _log_ds_info_accuracy(model: learner_ensemble.EnsembleLearner,
                          ds_info: ClassificationResults) -> NoReturn:
    r""" Log the base classifier performance results """
    # base_prefix = f"{model.name()} {ds_info.name}"
    base_prefix = f"{ds_info.name}"
    acc_flds = (
        ("Aggregated", ds_info.y),
        ("Original Labels", ds_info.lbls),
    )
    for desc, y in acc_flds:
        str_prefix = f"{base_prefix} {desc}:"

        logging.debug(f"{str_prefix} Dataset Size: {y.numel()}")
        # Pre-calculate fields needed in other calculations
        conf_matrix = pycm.ConfusionMatrix(actual_vector=y.numpy(),
                                           predict_vector=ds_info.y_pred.numpy())

        # noinspection PyUnresolvedReferences
        logging.debug(f"{str_prefix} Accuracy: {100. * conf_matrix.Overall_ACC:.3}%")

        for f1_type in ("Macro", "Micro"):
            f1 = conf_matrix.__getattribute__(f"F1_{f1_type}")
            logging.debug(f"{str_prefix} {f1_type} F1-Score: {f1:.6f}")

        # Write confusion matrix to a string so it can be logged
        sys.stdout = cm_out = io.StringIO()
        conf_matrix.print_matrix()
        sys.stdout = sys.__stdout__
        # Log the confusion matrix
        cm_str = cm_out.getvalue()
        logging.debug(f"{str_prefix} Confusion Matrix: \n{cm_str}")


def _log_ds_bounds(model: learner_ensemble.EnsembleLearner,
                   ds_info: ClassificationResults) -> float:
    r""" Log the robustness bound """
    prefix = utils.build_bound_prefix(desc="reg", model=model, bound_dist="NA",
                                      ds_name=ds_info.name, use_greedy_flag=config.USE_GREEDY)
    path = parent_utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")

    # Bounds may take a while to compute
    if not path.exists():
        bounds_res = []
        for true_lbl, lbl_desc in [(ds_utils.POS_LABEL, "pos"), (ds_utils.NEG_LABEL, "neg")]:
            mask = ds_info.y == true_lbl
            if len(mask.shape) > 1:
                mask = mask.squeeze(dim=1)
            assert len(mask.shape) == 1, "Unexpected shape of mask"

            # Target value is 0 so can just use the raw yhat values
            full_robust_dist = model.get_class_threshold() - ds_info.yhat
            lbound_val = ubound_val = None
            lbound_dist = ubound_dist = None
            if true_lbl == ds_utils.POS_LABEL:
                lbound_dist = full_robust_dist[mask].clip(max=0)
                lbound_val = model.get_class_threshold()
            else:
                ubound_dist = full_robust_dist[mask].clip(min=0)
                ubound_val = model.get_class_threshold()

            coverage = None
            if model.is_multicover():
                if lbound_val is not None:
                    lbound_val = torch.full_like(lbound_dist, lbound_val)
                if ubound_val is not None:
                    ubound_val = torch.full_like(ubound_dist, ubound_val)
                coverage = utils.calc_coverage(desc=f"cls-{lbl_desc}", ds_name=ds_info.name,
                                               bound_dist="NA",
                                               lbound=lbound_val, ubound=ubound_val,
                                               model=model, x=ds_info.x[mask])

            # Calculate the robustness bounds
            bound_res = model.calc_bound(full_yhat=ds_info.full_yhat[mask],
                                         ids=ds_info.ids[mask], l_cutoff_dist=lbound_dist,
                                         u_cutoff_dist=ubound_dist, coverage=coverage)

            bd_info = BoundInfo(true_lbl=true_lbl, mask=mask, dist=full_robust_dist,
                                bound_res=bound_res)
            bounds_res.append(bd_info)

        with open(path, "wb+") as f_out:
            pk.dump(bounds_res, f_out)

    with open(path, "rb") as f_in:
        bounds_res = pk.load(f_in)

    # Construct the bound info for both classes combined
    # noinspection PyTypeChecker
    bound = torch.full_like(ds_info.lbls, -1)  # type: LongTensor
    grb_info = []
    for bd_info in bounds_res:  # type: BoundInfo
        _cls_bound, _cls_grb_info = utils.extract_bound_results(bd_info.bound_res)
        bound[bd_info.mask] = _cls_bound
        if _cls_grb_info is not None:
            grb_info += _cls_grb_info

    # Construct the error distance.  Using a base threshold of passing of zero. That requires
    # negating the distance for the positive label since the bound distance for the positive
    # class is negative.  See above.
    err = torch.full_like(ds_info.y, -sys.maxsize, dtype=ds_info.full_yhat.dtype)
    for bd_info in bounds_res:  # type: BoundInfo
        full_dist = bd_info.dist[bd_info.mask]
        # Negate the true label
        if bd_info.true_lbl == ds_utils.NEG_LABEL:
            full_dist *= -1
        err[bd_info.mask] = full_dist
    min_err = -torch.min(err)

    # As explained above, bound distance is 0 which represents the example is correctly labeled.
    cert_ratio = utils.log_certification_ratio(model=model, err=err + min_err, y=ds_info.y,
                                               bound=bound, bound_dist=min_err.item(),
                                               override_bound_str="NA")

    return cert_ratio
