__all__ = [
    "build_bound_str",
    "build_bound_prefix",
    "calc_coverage",
    "extract_bound_results",
    "get_bound_log_width",
    "get_test_idx_to_analyze",
    "log_certification_ratio",
    "print_results",
]

import dill as pk
import logging
from typing import NoReturn, Optional, Union

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config
from .. import dirs
from .. import learner_ensemble
from ..types import Coverage, GurobiResList, TensorGroup
from .. import utils as parent_utils

# Number of test examples to analyze for most datasets
BASE_TE_COUNT = 200


def print_results(header: str, vals: Tensor, bin_width: float,
                  use_abs: bool, tot_count: Optional[int] = None) -> NoReturn:
    r"""
    Prints a simple perturbation histogram to understand the extent of each perturbation
    :param header: Header text before logging a result
    :param vals:
    :param bin_width: Width of the bin when printing the histogram
    :param use_abs: If \p True, dataset contains negative data so print with absolute value
    :param tot_count: Allows for optional logging of a different bound fraction
    """
    assert bin_width > 0, "Bin width must be positive"

    # Half the number of bins to be used
    half_bins = vals.abs().max().div(bin_width).ceil()
    n_bins = half_bins.long()
    if not use_abs:
        max_window, _ = torch.max(vals.max().div(bin_width).ceil(), 0)
        min_window, _ = torch.min(vals.min().div(bin_width).floor(), 0)
    else:
        max_window = half_bins * bin_width
        min_window = -max_window
        n_bins *= 2
    if vals.dtype == torch.long:
        vals = vals.float()
    if tot_count is None:
        tot_count = vals.numel()
    # Window is based on the bin width
    min_window, max_window = bin_width * min_window, max_window * bin_width

    if n_bins == 0:
        logging.warning(f"{header}: No bins exist. Skipping print.")
        return

    # Allow for small floating point errors
    # noinspection PyTypeChecker
    histc = torch.histc(vals, min=min_window, max=max_window, bins=n_bins)
    histc = histc.long()  # type: LongTensor
    assert vals.numel() == torch.sum(histc).item(), "Elements lost in histogram"

    # Number of digits when logging the bins
    n_dig = 1 if isinstance(bin_width, float) else 0

    logging.info(f"{header} Dataset Size: {vals.numel()}")
    logging.info(f"{header} Bound Fraction Count: {tot_count}")

    lin = torch.linspace(min_window, max_window, steps=n_bins + 1)
    # Define the width of the bin boundaries to allow text to line up when logging
    txt_width = max(len(f"{val.item():.{n_dig}f}") for val in [lin[0], lin[-1]])

    # Exclude anything that exceeds min_val or max_val
    if histc.numel() > n_bins:
        logging.info(f"{header} Other Excluded: {histc[-1].item()}")

    is_abs_vals = [False]
    if use_abs:
        is_abs_vals.append(True)
    # Only log the absolute value separately if there are negative values
    for is_abs in is_abs_vals:
        desc = "Abs" if is_abs else "Raw"
        tmp_header = f"{header} ({desc})"
        # Print perturbation stats
        # noinspection DuplicatedCode
        quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.])
        names = ["Min", "25%-Quartile", "Median", "75%-Quartile", "Max"]
        quant_vals = torch.quantile(vals, q=quantiles)
        for name, val in zip(names, quant_vals.tolist()):
            logging.info(f"{tmp_header} Perturb {name}: {val:.3f}")
        # Interquartile range
        val = quant_vals[-2] - quant_vals[1]
        logging.info(f"{tmp_header} IQR: {val.item():.3f}")

        std, mean = torch.std_mean(vals, unbiased=True)
        for val, val_name in zip((mean, std), ("Mean", "Stdev")):
            logging.info(f"{tmp_header} {val_name}: {val.item():.3f}")


def log_certification_ratio(model: learner_ensemble.EnsembleLearner, err: Tensor,
                            y: Tensor, bound: LongTensor, bound_dist: float,
                            override_bound_str: str = "") -> float:
    r""" Log the certification ratio for the predictions """
    assert err.numel() == bound.numel(), "Mismatch in length error and bounds"

    if len(bound.shape) > 1:
        bound = bound.squeeze(dim=1)
    assert len(bound.shape) == 1, "Unexpected size of the bound tensors"

    submodel_cls = model.get_submodel_type()

    bound_str = build_bound_str(bound_dist=bound_dist)
    if override_bound_str:
        bound_str = override_bound_str
    header = f"{model.name()} (d={bound_str})"

    bound_val = calc_bound_val(dist=bound_dist, y=y)

    # Only consider those examples within the correctness range
    # noinspection PyUnresolvedReferences
    correct_mask = (err > -bound_val).logical_and(err < bound_val)
    _print_cert_res(header=header, model=model, submodel_cls=submodel_cls, mask=correct_mask,
                    bound_cnt=0, bound_dist=bound_dist, override_bound_str=override_bound_str)

    step = get_bound_log_width()
    for i in range(step, bound.max().item() + 1, step):
        bound_mask = bound >= i

        # Certified if both (sufficiently) correct and certified
        # assert correct_mask.shape == bound_mask.shape, "Mismatch match in shape of masks"
        mask = correct_mask.logical_and(bound_mask)
        bound_dist_log_val = bound_dist if not override_bound_str else override_bound_str
        _print_cert_res(header=header, model=model, submodel_cls=submodel_cls, mask=mask,
                        bound_cnt=i, bound_dist=bound_dist_log_val,
                        override_bound_str=override_bound_str)

    # Baseline certification ratio
    base_cert = correct_mask.logical_and(bound >= 0)
    tot_cert = torch.sum(base_cert).item()
    cert_ratio = tot_cert / bound.numel()
    return cert_ratio


def _print_cert_res(header: str, model: learner_ensemble.EnsembleLearner, mask: BoolTensor,
                    submodel_cls, bound_cnt: int, bound_dist: Optional[float],
                    override_bound_str: str = "") -> NoReturn:
    r""" Standardizes printing the certification results """
    tot_count = mask.numel()

    cert_count = torch.sum(mask).item()
    # logging.info(f"{header} Cert. Model Uses Alt Submodel: {model.uses_alt}")
    # cls_name = model.__class__.__name__
    # logging.info(f"{header} Cert. Model Class: {cls_name}")
    logging.info(f"{header} Cert. Model Type: {model.cover_type}")
    logging.info(f"{header} Cert. Model # Submodels: {model.n_models}")
    logging.info(f"{header} Cert. Submodel Type: {submodel_cls.__name__}")
    logging.info(f"{header} Cert. Model PPM: {model.ppm}")

    bound_str = override_bound_str
    if not bound_str:
        bound_str = build_bound_str(bound_dist=bound_dist)
    logging.info(f"{header} Cert. Dist: {bound_str}")
    logging.info(f"{header} Cert. Bound: {bound_cnt}")

    ratio = cert_count / tot_count
    logging.info(f"{header} Cert. Count: {cert_count} / {tot_count} ({ratio:.2%})")


def build_bound_prefix(desc: str, model: learner_ensemble.EnsembleLearner, ds_name: str,
                       bound_dist: Optional[Union[float, int, str]],
                       use_greedy_flag: bool = False) -> str:
    r""" Construct the results prefix for primary bound results """
    return _base_build_res_prefix(desc=desc, base_desc="bd-info", model=model, ds_name=ds_name,
                                  bound_dist=bound_dist, use_greedy_flag=use_greedy_flag)


def _base_build_res_prefix(desc: str, base_desc: str, model: learner_ensemble.EnsembleLearner,
                           ds_name: str, bound_dist: Optional[Union[float, int, str]],
                           include_grb_timeout: bool = True,
                           use_greedy_flag: bool = False) -> str:
    r""" Standardizes building the results filename prefix """
    # Include the bound distance in the name so it is possible to test multiple bounsd
    flds = [
        desc,
        base_desc,
        model.name(),
        ds_name,
        build_bound_str(bound_dist=bound_dist),
    ]
    if include_grb_timeout and parent_utils.include_gurobi_timeout_field():
        flds.append(parent_utils.build_grb_timeout_fld())
    if config.IS_NO_MULTICOVER:
        flds.append("override")
    if use_greedy_flag and config.USE_GREEDY:
        flds.append("greedy")
    return "_".join(flds).lower().replace(" ", "-")


def calc_coverage(desc: str, model: learner_ensemble.EnsembleLearner, ds_name: str,
                  bound_dist: Union[str, float], x: Tensor,
                  lbound: Optional[Tensor], ubound: Optional[Tensor]) -> Optional[Coverage]:
    r"""
    Calculates and returns the coverage for the specified model.
    :param desc: Description of the type of results being returned
    :param model: Ensemble model being analyzed
    :param ds_name: Name of the dataset being analyzed
    :param bound_dist: Overall base distance being analyzed
    :param x: Features tensor to calculate coverage over
    :param lbound:
    :param ubound:
    :return: Coverage quantity on a per submodel basis
    """
    assert model.is_multicover(), "Coverage only applies for multicover models"
    if config.IS_NO_MULTICOVER:
        return None

    prefix = _base_build_res_prefix(desc=desc, base_desc="coverage", model=model,
                                    bound_dist=bound_dist, ds_name=ds_name,
                                    include_grb_timeout=False)
    path = parent_utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")

    if not path.exists():
        msg = "Calculating coverage"
        logging.info(f"Starting: {msg}...")
        coverage = model.calc_coverage(x=x, lbound=lbound, ubound=ubound)
        with open(path, "wb+") as f_out:
            pk.dump(coverage, f_out)
        logging.info(f"COMPLETED: {msg}")

    with open(path, "rb") as f_in:
        coverage = pk.load(f_in)

    for cover in [coverage.l_cover, coverage.u_cover]:
        if cover is None:
            continue
        assert cover.shape[0] == x.shape[0], "Coverage does not include expected number of examples"
        assert cover.shape[1] == model.n_models, "Coverage does not include all submodels"

    return coverage


def build_bound_str(bound_dist: Union[float, int, str]) -> str:
    r""" Construct the bound string from the distance """
    assert isinstance(bound_dist, (float, int, str)), f"Type is {bound_dist.__class__.__name__}"
    bound_str = str(bound_dist)
    if config.IS_BOUND_PERCENT:
        bound_str += "%"
    return bound_str


def calc_bound_val(dist: float, y: Tensor) -> Union[float, Tensor]:
    r""" Standardizes the calculation of the bound value """
    if not config.IS_BOUND_PERCENT:
        return dist
    return dist / 100 * y


def get_bound_log_width() -> int:
    r""" Gets the histogram bin width for logging bounds """
    return 1


def extract_bound_results(bound_res) -> GurobiResList:
    r""" Parse the bound results including handling the old results format """
    bound_vals, grb_info = bound_res
    # noinspection PyTypeChecker
    return bound_vals, grb_info


def get_test_idx_to_analyze(tg: TensorGroup) -> LongTensor:
    r""" Accessor for the number of test examples to analyze """
    n_test = tg.test_y.numel()  # Test set's size

    n_analyze = BASE_TE_COUNT
    n_keep = min(n_test, n_analyze)

    keep_idx = torch.randperm(n_test, dtype=torch.long)[:n_keep]
    return keep_idx
