__name__ = [
    "calc_multi",
    "calc_single",
]

import logging
import sys
from typing import Optional

import gurobipy as grb
from gurobipy import GRB
import torch
from torch import LongTensor

from .types import GurobiResult
from .. import logger
from .. import _config as config


def calc_single(model, ex_id: int, is_lower: bool, perturb_dist: int, models_list: LongTensor,
                best_bd_stop: Optional[int], relax: bool = False) -> GurobiResult:
    r"""
    Performs partial set (single) cover using Gurobi.

    :param model: Ensemble learner object
    :param ex_id: ID number of the (test) example being analyzed.
    :param is_lower: If \p True, analyzing the lower bound of the regression limit.  This is
                     exclusively used for logging and does not affect the actual optimization.
    :param perturb_dist: Number of models to perturb
    :param models_list: List of models that can be perturbed to affect the prediction
    :param best_bd_stop: If specified, allows for early cutoff of the gurobi calculation since
                         there is no point optimizing one bound beyond the best bound of the
                         side (if applicable)
    :param relax: If \p True, perform the analysis on a relaxed (i.e., non-integral) version of
                  the ILP.
    :return: Gurobi execution results object.
    """
    return _calc_wrapper(model=model, ex_id=ex_id, is_lower=is_lower, perturb_dist=perturb_dist,
                         models_list=models_list, coverage=None, best_bd_stop=best_bd_stop,
                         relax=relax)


def calc_multi(model, ex_id: int, is_lower: bool, perturb_dist: int, models_list: LongTensor,
               coverage: LongTensor, best_bd_stop: Optional[int],
               relax: bool = False) -> GurobiResult:
    r"""
    Performs partial set multi-cover using Gurobi.

    :param model: Ensemble learner object
    :param ex_id: ID number of the (test) example being analyzed.
    :param is_lower: If \p True, analyzing the lower bound of the regression limit.  This is
                     exclusively used for logging and does not affect the actual optimization.
    :param perturb_dist: Number of models to perturb
    :param models_list: List of models that can be perturbed to affect the prediction
    :param coverage: Number of training instances within each model in \p models-list must be
                     perturbed for the model to sufficiently change the prediction
    :param best_bd_stop: If specified, allows for early cutoff of the gurobi calculation since
                         there is no point optimizing one bound beyond the best bound of the
                         side (if applicable)
    :param relax: If \p True, perform the analysis on a relaxed (i.e., non-integral) version of
                  the ILP.
    :return: Gurobi execution results object.
    """
    assert coverage.numel() == models_list.numel(), "Mismatch in number of models"
    assert torch.all(coverage > 0), "Negative coverage is not supported"

    return _calc_wrapper(model=model, ex_id=ex_id, is_lower=is_lower, perturb_dist=perturb_dist,
                         models_list=models_list, coverage=coverage, best_bd_stop=best_bd_stop,
                         relax=relax)


def _calc_wrapper(model, ex_id: int, is_lower: bool, perturb_dist: int, models_list: LongTensor,
                  coverage: Optional[LongTensor], best_bd_stop: Optional[int],
                  relax: bool) -> GurobiResult:
    r"""
    Wraps the main calculation of base calc since stdout re-routing is used

    :param model: Ensemble learner object
    :param ex_id: ID number of the (test) example being analyzed.
    :param is_lower: If \p True, analyzing the lower bound of the regression limit.  This is
                     exclusively used for logging and does not affect the actual optimization.
    :param perturb_dist: Number of models to perturb
    :param models_list: List of models that can be perturbed to affect the prediction
    :param best_bd_stop: If specified, allows for early cutoff of the gurobi calculation since
                         there is no point optimizing one bound beyond the best bound of the
                         side (if applicable)
    :param relax: If \p True, perform the analysis on a relaxed (i.e., non-integral) version of
                  the ILP.
    :return: Gurobi execution results object.
    """
    class DevNull:
        def write(self, *args, **kwargs):
            pass

        def flush(self, *args, **kwargs):
            pass

    # Disable stdout to prevent double logging when using the logging package
    sys.stdout = DevNull()

    res = _calc_main(model=model, ex_id=ex_id, perturb_dist=perturb_dist, is_lower=is_lower,
                     models_list=models_list, coverage=coverage, best_bd_stop=best_bd_stop,
                     relax=relax)
    try:
        return res
    except Exception:
        # restore stdout so that handlers can print normally
        # https://docs.python.org/3/library/sys.html#sys.__stdout__
        sys.stdout = sys.__stdout__
        raise
    finally:
        sys.stdout = sys.__stdout__


def _calc_main(model, ex_id: int, is_lower: bool, perturb_dist: int, models_list: LongTensor,
               coverage: Optional[LongTensor], best_bd_stop: Optional[int],
               relax: bool) -> GurobiResult:
    r"""
    Calculates both single and multi partial cover.

    :param model: Ensemble learner object
    :param ex_id: ID number of the (test) example being analyzed.
    :param is_lower: If \p True, analyzing the lower bound of the regression limit.  This is
                     exclusively used for logging and does not affect the actual optimization.
    :param perturb_dist: Number of models to perturb
    :param models_list: List of models that can be perturbed to affect the prediction
    :param best_bd_stop: If specified, allows for early cutoff of the gurobi calculation since
                         there is no point optimizing one bound beyond the best bound of the
                         side (if applicable)
    :param relax: If \p True, perform the analysis on a relaxed (i.e., non-integral) version of
                  the ILP.
    :return: Gurobi execution results object.
    """
    is_single = coverage is None  # True denotes single cover is used
    if coverage is not None and torch.all(coverage == 1):
        logging.info("Reverting to single cover since all single cover values")
        is_single = True

    desc = "single" if is_single else "multi"

    # Name of the model
    flds = [
        f"partial-{desc}-cover",
        model.name(),
        f"id={ex_id}",
        "lower" if is_lower else "upper",
        "relax" if relax else "integral",
    ]
    if config.IS_NO_MULTICOVER:
        flds.append("override")
    name = "_".join(flds)

    logging.debug(f"***************  Starting Gurobi for {name}  ***************")
    ilp = grb.Model(name)
    ilp.setParam("Threads", logger.get_num_usable_cpus())

    if best_bd_stop is not None:
        # Gurobi recommends including a "small tolerance" in this value to prevent rounding
        # errors causing a premature stop. See
        # https://www.gurobi.com/documentation/9.5/refman/bestbdstop.html
        bd = best_bd_stop + 1e-4
        ilp.setParam("BestBdStop", bd)

    if config.has_gurobi_timeout():
        ilp.setParam('TimeLimit', config.GUROBI_TIMEOUT)

    # Exact approach uses integer vars. Relaxed solution uses continuous variables
    if not relax:
        mod_kwargs = ds_kwargs = {"vtype": GRB.BINARY}
        if not is_single:
            # No integer can exceed the maximum coverage so set as an upper bound on the variables
            max_cover = torch.max(coverage).item()
            # Datasets may need to be perturbed many times for multicover so use integer values
            # but only for the datasets as a model can only be perturbed once
            ds_kwargs = {"vtype": GRB.INTEGER, "lb": 0, "ub": max_cover}
            # There can be a mismatch between the reported bound and the best objective value
            # due to numerical instability. The setting below improves the logging performance
            # to prevent those issues.  See the two links below:
            # https://support.gurobi.com/hc/en-us/community/posts/360077289292-Nonzero-gap-for-optimal-solution-
            # https://www.gurobi.com/documentation/9.1/refman/numericfocus.html
            ilp.setParam("NumericFocus", 3)
            ilp.setParam("Presolve", 2)
    else:
        # For continuous variables, also specify the range of the variables
        mod_kwargs = ds_kwargs = {"vtype": GRB.CONTINUOUS, "lb": 0, "ub": 1}
        if not is_single:
            ds_kwargs = {"vtype": GRB.CONTINUOUS, "lb": 0}

    ds_perturb = ilp.addVars(model.n_ds_parts, name="Ds-Part-Is-Perturb", **ds_kwargs)
    # Only consider those models which if perturbed matter.  All others do not matter.
    # model_cover = ilp.addVars(model.n_models, name=f"Model-Is-Covered", **kwargs)
    model_cover = ilp.addVars(len(models_list), name=f"Model-Is-Covered", **mod_kwargs)

    # Objective minimizes the number of dataset parts that are perturbed
    objective = ds_perturb.sum()

    # Math is slightly different when doing multi-cover. Correct the math by adjusting the
    # perturbation distance and objective values
    if not is_single:
        # Perturbation distance figures differently in multicover. In multicover, a model
        # is not perturbed until coverage is reached so the true bound is one less than the
        # cover value
        perturb_dist += 1
        # Since the perturbation distance is increased by 1 for multicover, need to decrement
        # the bound by 1 since cannot reach the full perturbation distance
        objective -= 1

    # Short circuit if the result is no bound
    if perturb_dist == 0:
        return GurobiResult.create_noop(val=perturb_dist, ex_id=ex_id, is_lower=is_lower,
                                        is_single=is_single)

    ilp.setObjective(objective, GRB.MINIMIZE)
    # Ensure sufficient models perturbed
    ilp.addConstr(model_cover.sum() >= perturb_dist, "Set-Cover-Constraint")

    # Creates the set cover constraints.
    # mod_idx is a simple index to connected model ID numbers to underlying variables in
    # the ILP.
    sorted_models_list = sorted(models_list.tolist())
    for mod_idx, model_id in enumerate(sorted_models_list):
        ds_parts_lst = model.get_submodel_ds_parts(model_id)
        assert len(ds_parts_lst) == config.SPREAD_DEGREE, "Unexpected dataset parts length"

        map_sum = grb.quicksum(ds_perturb[subs_idx] for subs_idx in ds_parts_lst)

        constraint_name = f"cov-constraint-{model_id:05d}"
        # May not need to break out into two different commands for single and multi-cover.
        # Unsure how an unnecessary multiplicative factor may affect Gurobi's performance
        # if that factor is 1.
        if is_single:
            ilp.addConstr(model_cover[mod_idx] <= map_sum, constraint_name)
        else:
            cover_i = coverage[mod_idx].item()
            ilp.addConstr(cover_i * model_cover[mod_idx] <= map_sum, constraint_name)

    # Perform the optimization
    ilp.update()
    ilp.optimize()
    # ilp.write("ilp.mps")

    # Report the results
    flds = [
        model.name(),
        f"Ex={ex_id}",
        "Lower" if is_lower else "Upper",
        "Bound",
        "Relaxed" if relax else "Integral",
    ]
    header = " ".join(flds)

    res = GurobiResult.create(ilp=ilp, relax=relax, ex_id=ex_id, is_lower=is_lower,
                              is_single=is_single)
    res.log(header=header, is_debug=True)

    return res
