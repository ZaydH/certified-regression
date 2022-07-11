__all__ = [
    "MultiCoverEnsemble",
    "SingleCoverEnsemble",
    "train_multi_cover_ensemble",
    "train_single_cover_ensemble",
]

import abc
import dill as pk
import logging
import time
from typing import List, Optional, Union

import torch
from torch import Tensor, LongTensor

from . import _config as config
from . import _greedy_bound as greedy_bound
from . import dirs
from .gurobi_utils import GurobiResult, partial_cover
from . import learner_ensemble
from .types import BoundReturnType, Coverage, TensorGroup
from . import utils


class _BaseCoverEnsemble(learner_ensemble.EnsembleLearner, abc.ABC):
    r""" Base model for single and multi-cover models """
    def __init__(self, n_models: int, spread_degree: int, opt_params: Optional[dict] = None):
        n_parts = n_models
        model_ds_parts = self._assign_ds_parts(n_models=n_models, spread_degree=spread_degree)

        flds = []
        if config.is_alt_submodel():
            flds.append(config.ALT_TYPE.name.lower())
        flds += [self.cover_type, f"n-mod={n_models:04d}", f"ppm={spread_degree}"]

        super().__init__(prefix="-".join(flds), n_ds_parts=n_parts, model_ds_parts=model_ds_parts,
                         opt_params=opt_params)

    @staticmethod
    def _assign_ds_parts(n_models: int, spread_degree: int) -> List[List[int]]:
        r"""
        Assigns dataset parts used to train each of the models

        The implementation does not directly rely on a hash function.  Instead it performs
        a series of random permutations.  This ensures that each model is trained on the same
        number of dataset partitions.

        :param n_models: Number of models in the ensemble
        :param spread_degree: Spreading degree of each training set part
        :return: Assignment of dataset parts to the models
        """
        logging.debug("Reseeding the random number generator for consistent part splitting")
        utils.set_random_seeds()

        model_ds_parts = [[] for _ in range(n_models)]
        part_lst = torch.arange(n_models, dtype=torch.long)
        for _ in range(spread_degree):
            part_lst = part_lst[torch.randperm(n_models)]
            # Select the dataset parts assigned to each of the ensemble models
            for part_id, model_id in enumerate(part_lst.tolist()):
                # noinspection PyTypeChecker
                model_ds_parts[model_id].append(part_id)
        return model_ds_parts

    def _calc_bound(self, full_yhat: Tensor, ids: LongTensor,
                    l_cutoff_dist: Optional[Union[float, Tensor]],
                    u_cutoff_dist: Optional[Union[float, Tensor]],
                    coverage: Coverage) -> BoundReturnType:
        r""" Calculates the regression perturbation bound """
        # raise NotImplementedError
        assert full_yhat.shape[1] == self.n_models, "Mismatch between num results and num models"
        assert full_yhat.shape[0] == ids.numel(), "Mismatch in number of examples"
        assert len(ids.shape) == 1 or len(ids.shape) == 2 and ids.shape[1], \
            f"Bizarre IDs shape {ids.shape}"

        # Perturbation distance needed to change a prediction either higher or lower
        l_perturb_dist, u_perturb_dist = self._calc_bound_dists(full_yhat=full_yhat,
                                                                lbound_change=l_cutoff_dist,
                                                                ubound_change=u_cutoff_dist)
        # Determine whether low-side bound or high-side bound will be calculated below
        skip_ubound = skip_lbound = True
        if l_cutoff_dist is not None:
            skip_lbound = False
            assert l_cutoff_dist is not None, "Expected a low-side bound distance"
            # Arguments specific to low-side bound calculation
            lbound_kwargs = {"l_cutoff_dist": l_cutoff_dist, "l_perturb_dist": l_perturb_dist}
        if u_cutoff_dist is not None:
            skip_ubound = False
            assert u_cutoff_dist is not None, "Expected a high-side bound distance"
            # Arguments specific to high-side bound calculation
            ubound_kwargs = {"u_cutoff_dist": u_cutoff_dist, "u_perturb_dist": u_perturb_dist}
        # If determining both low and high side belows determine which side to run first
        run_lbound_first = torch.full(ids.shape, not skip_lbound, dtype=torch.bool)
        if not skip_lbound and not skip_ubound:
            run_lbound_first = l_perturb_dist <= u_perturb_dist
            assert run_lbound_first.shape == l_perturb_dist.shape, "Unexpected bound shape"

        # Sort the predictions for each test example in ascending order. This is needed to
        # determine which models (and by extension datasets) should be perturbed.
        sort_full_yhat, sort_idx = full_yhat.sort(dim=1)
        half_pt = self._med_model_cnt()

        bound_vals, grb_info = [], []
        for i_ele in range(full_yhat.shape[0]):
            msg = f"Analyzing bound instance {i_ele + 1} of {full_yhat.shape[0]}"
            logging.info(f"Starting: {msg}")

            ele_yhat, ele_med = sort_full_yhat[i_ele], sort_full_yhat[i_ele, half_pt]

            # Base arguments shared by all bound calculation functions
            base_kwargs = {"ex_id": ids[i_ele].item(), "i_ele": i_ele,
                           "ele_sort_idx": sort_idx[i_ele],
                           "ele_yhat": ele_yhat, "ele_ymed": ele_med,
                           "coverage": coverage}

            lbound = ubound = None
            # best_bd_stop allows for early stopping of either the bound calculation. To prevent
            # unnecessary time spent calculating a bound, run which bound first has a smaller
            # perturbation distance. From there, use the best bound from the first step to
            # allow potentially early stopping on the other bound.
            if skip_ubound or run_lbound_first[i_ele]:
                # noinspection PyUnboundLocalVariable
                lbound = self._calc_low_limit_bound(best_bd_stop=None,
                                                    **lbound_kwargs, **base_kwargs)
                if not skip_ubound:
                    # noinspection PyUnboundLocalVariable
                    ubound = self._calc_high_limit_bound(best_bd_stop=lbound.low_bound,
                                                         **ubound_kwargs, **base_kwargs)
            else:
                ubound = self._calc_high_limit_bound(best_bd_stop=None,
                                                     **ubound_kwargs, **base_kwargs)
                if not skip_lbound:
                    lbound = self._calc_low_limit_bound(best_bd_stop=ubound.low_bound,
                                                        **lbound_kwargs, **base_kwargs)

            # Store the bound information
            grb_info.append((lbound, ubound))
            # Construct the bound value
            vals = [bd_val.low_bound for bd_val in (lbound, ubound) if bd_val is not None]
            bound_vals.append(min(vals))

            logging.info(f"COMPLETED: {msg}")
        # Aggregate all the bounds
        # noinspection PyTypeChecker
        joint_bounds = torch.tensor(bound_vals, dtype=torch.long)  # type: LongTensor
        return joint_bounds, grb_info

    def _calc_low_limit_bound(self, ex_id: int, i_ele: int,
                              l_cutoff_dist: Tensor, l_perturb_dist: LongTensor,
                              ele_yhat: Tensor, ele_ymed: Tensor, ele_sort_idx: LongTensor,
                              best_bd_stop: Optional[int],
                              coverage: Coverage) -> GurobiResult:
        r"""
        Calculates the low-side bound.

        :param ex_id: Example ID number
        :param i_ele: Index of the element of interest.
        :param l_cutoff_dist: Distance to the low-side cutoff
        :param l_perturb_dist:
        :param ele_yhat:
        :param ele_ymed:
        :param ele_sort_idx:
        :param best_bd_stop: Best bound to early stop the ILP.  If not specified, ILP runs to
                             optimal or timeout.
        :return: Low-side certification bound
        """
        low_idx = self._med_model_cnt() - l_perturb_dist[i_ele]

        l_cover = None
        if coverage is not None and coverage.l_cover is not None:
            l_cover = coverage.l_cover[i_ele]

        # Error check that the low cutoff point is correct
        dist = l_cutoff_dist[i_ele]
        # Error dist is a negative quantity for lower bound so still use addition below
        assert low_idx == self.n_models or ele_yhat[low_idx] > ele_ymed + dist, \
            "Val above min cut"
        assert low_idx == 0 or ele_yhat[low_idx - 1] <= ele_ymed + dist, "Val above cut"

        # No need to add 1 since low_idx is included when slicing from the bottom
        lower_models = ele_sort_idx[low_idx:]
        perturb_dist = l_perturb_dist[i_ele].item()
        if config.USE_GREEDY:
            return self._calc_greedy_cover(model_ids=lower_models, perturb_dist=perturb_dist,
                                           is_lower=True, ex_id=ex_id)
        else:
            return self._calc_true_bound(model_ids=lower_models, ex_id=ex_id,
                                         perturb_dist=perturb_dist, is_lower=True,
                                         best_bd_stop=best_bd_stop, coverage=l_cover)

    def _calc_high_limit_bound(self, ex_id: int, i_ele: int,
                               u_cutoff_dist: Tensor, u_perturb_dist: LongTensor,
                               ele_yhat: Tensor, ele_ymed: Tensor, ele_sort_idx: LongTensor,
                               best_bd_stop: Optional[int],
                               coverage: Coverage) -> GurobiResult:
        r"""
        Calculates the high-side bound

        :param ex_id: Example ID number
        :param i_ele: Index of the element of interest.
        :param u_cutoff_dist:
        :param u_perturb_dist:
        :param ele_yhat:
        :param ele_ymed:
        :param ele_sort_idx:
        :param best_bd_stop: Best bound to early stop the ILP.  If not specified, ILP runs to
                             optimal or timeout.
        :return: High-side certification bound
        """
        dist = u_cutoff_dist[i_ele]

        u_cover = None
        if coverage is not None and coverage.u_cover is not None:
            u_cover = coverage.u_cover[i_ele]

        hi_idx = self._med_model_cnt() + u_perturb_dist[i_ele]
        # Error check that the high cutoff point is correct

        assert hi_idx == -1 or ele_yhat[hi_idx] < ele_ymed + dist, "Val above max cut"
        assert hi_idx + 1 == self.n_models or ele_yhat[hi_idx + 1] >= ele_ymed + dist, \
            "Val below cut"
        # Add one so the hi_idx is included
        upper_models = ele_sort_idx[:hi_idx + 1]
        perturb_dist = u_perturb_dist[i_ele].item()

        if config.USE_GREEDY:
            assert not self.is_multicover(), "Greedy bound calc. not supported with multicover"
            return self._calc_greedy_cover(model_ids=upper_models, perturb_dist=perturb_dist,
                                           is_lower=False, ex_id=ex_id)
        else:
            return self._calc_true_bound(model_ids=upper_models, ex_id=ex_id,
                                         perturb_dist=perturb_dist, is_lower=False,
                                         best_bd_stop=best_bd_stop, coverage=u_cover)

    def _calc_true_bound(self, ex_id: int, is_lower: bool, model_ids: LongTensor,
                         perturb_dist: int, coverage: Optional[LongTensor],
                         best_bd_stop: Optional[int]) -> GurobiResult:
        r""" Implements the call to the coverage function """
        # if perturb_dist <= 0:
        #     return GurobiResult.create_noop(val=perturb_dist, ex_id=ex_id, is_lower=is_lower,
        #                                     is_single=coverage is None)

        # Gurobi can take a while so serialize the temp results
        flds = [
            self.name(),
            f"ex={ex_id:06d}",
            f"is_low={is_lower}",
            f"d={perturb_dist}",
        ]
        if utils.include_gurobi_timeout_field():
            flds.append(utils.build_grb_timeout_fld())
        if config.IS_NO_MULTICOVER:
            flds.append("override")
        res_dir = dirs.RES_DIR / self.name().lower()
        res_dir.mkdir(exist_ok=True, parents=True)
        serialize_path = utils.construct_filename(prefix="-".join(flds), out_dir=res_dir,
                                                  file_ext="pkl")

        if not serialize_path.exists():
            res = self._run_gurobi_cover(ex_id=ex_id, is_lower=is_lower, model_ids=model_ids,
                                         perturb_dist=perturb_dist, coverage=coverage,
                                         best_bd_stop=best_bd_stop)

            # Serialize the results
            with open(serialize_path, "wb+") as f_out:
                pk.dump(res, f_out)

        with open(serialize_path, "rb") as f_in:
            res = pk.load(f_in)
        return res

    @abc.abstractmethod
    def _run_gurobi_cover(self, ex_id: int, is_lower: bool, perturb_dist: int,
                          model_ids: LongTensor, coverage: Optional[LongTensor],
                          best_bd_stop: Optional[int]) -> GurobiResult:
        r""" Implements the Gurobi cover call """

    def _run_single_gurobi_cover(self, ex_id: int, is_lower: bool, perturb_dist: int,
                                 model_ids: LongTensor,
                                 best_bd_stop: Optional[int]) -> GurobiResult:
        r""" Performs partial set (single) cover using Gurobi """
        return partial_cover.calc_single(model=self, ex_id=ex_id, is_lower=is_lower,
                                         perturb_dist=perturb_dist, models_list=model_ids,
                                         best_bd_stop=best_bd_stop)

    def _calc_greedy_cover(self, ex_id: int, is_lower: bool, model_ids: LongTensor,
                           perturb_dist: int) -> GurobiResult:
        r"""
        Calculates the greedy set cover.  Model IDs represents the set of models which if perturbed
        would affect the median calculation.

        :param model_ids: ID number of the models which if perturbed affect the prediction
        :param perturb_dist: Number of models that must be perturbed to sufficiently affect the
            median
        :return: Robustness bound on the prediction
        """
        start_time = time.time()

        greedy_res = greedy_bound.calc(model=self, models_list=model_ids,
                                       perturb_dist=perturb_dist)

        end_time = time.time()

        return GurobiResult(ex_id=ex_id, is_relax=True, is_lower=is_lower,
                            low_bound=greedy_res.low_bound, runtime=end_time - start_time,
                            obj_val=greedy_res.greedy_val, timed_out=False, is_single=True)


class SingleCoverEnsemble(_BaseCoverEnsemble):
    r""" Basic single cover ensemble  """
    @property
    def cover_type(self) -> str:
        r""" Define the cover type of the model """
        return "single-cover"

    @staticmethod
    def is_multicover() -> bool:
        r""" Return \p True if the model supports multi-coverage """
        return False

    def _run_gurobi_cover(self, ex_id: int, is_lower: bool, perturb_dist: int,
                          model_ids: LongTensor, coverage: Optional[LongTensor],
                          best_bd_stop: Optional[int]) -> GurobiResult:
        r""" Performs partial set (single) cover using Gurobi """
        assert coverage is None, "Coverage is ignored in single cover"
        return self._run_single_gurobi_cover(ex_id=ex_id, is_lower=is_lower,
                                             perturb_dist=perturb_dist, model_ids=model_ids,
                                             best_bd_stop=best_bd_stop)


class MultiCoverEnsemble(_BaseCoverEnsemble):
    @property
    def cover_type(self) -> str:
        r""" Define the cover type of the model """
        return "multi-cover"

    @staticmethod
    def is_multicover() -> bool:
        r""" Return \p True if the model supports multi-coverage """
        return True

    def _run_gurobi_cover(self, ex_id: int, is_lower: bool, perturb_dist: int,
                          model_ids: LongTensor, coverage: Optional[LongTensor],
                          best_bd_stop: Optional[int]) -> GurobiResult:
        r""" Performs partial set multi-cover using Gurobi """
        assert coverage is not None or config.IS_NO_MULTICOVER, \
            "Coverage is expected in multicover but not specified"

        if not config.IS_NO_MULTICOVER:
            # Select the coverage values corresponding to the models be considered
            coverage = coverage[model_ids]

            res = partial_cover.calc_multi(model=self, ex_id=ex_id, is_lower=is_lower,
                                           perturb_dist=perturb_dist, models_list=model_ids,
                                           coverage=coverage, best_bd_stop=best_bd_stop)
        else:
            logging.warning("Overriding multicover mode using single cover")
            return self._run_single_gurobi_cover(ex_id=ex_id, is_lower=is_lower,
                                                 perturb_dist=perturb_dist, model_ids=model_ids,
                                                 best_bd_stop=best_bd_stop)
        return res


def train_single_cover_ensemble(tg: TensorGroup, opt_params: Optional[dict] = None) \
        -> SingleCoverEnsemble:
    r"""
    Train a set of ensemble models where each model

    :param tg: Tensor groups
    :param opt_params: Optional model parameters. Primarily used for hyperparameter tuning

    :return: Collection of trained classifiers
    """
    return _base_train_cover_ensemble(tg=tg, opt_params=opt_params, is_single=True)


def train_multi_cover_ensemble(tg: TensorGroup, opt_params: Optional[dict] = None) \
        -> MultiCoverEnsemble:
    r"""
    Train a set of disjoint models

    :param tg: Tensor groups
    :param opt_params: Optional model parameters. Primarily used for hyperparameter tuning

    :return: Collection of trained classifiers
    """
    return _base_train_cover_ensemble(tg=tg, opt_params=opt_params, is_single=False)


def _base_train_cover_ensemble(tg: TensorGroup, opt_params: Optional[dict], is_single: bool) \
        -> Union[SingleCoverEnsemble, MultiCoverEnsemble]:
    inv_train_frac = config.N_DISJOINT_MODELS  # k
    spread_degree = config.SPREAD_DEGREE  # d

    n_models = inv_train_frac * spread_degree  # DFA uses k * d models

    # Prefix defines information about the training set
    prefix_flds = []
    if config.is_alt_submodel():
        prefix_flds.append(config.ALT_TYPE.name.lower())
    cover_desc = "single" if is_single else "multi"
    prefix_flds += [
        f"{cover_desc}-cover",
        f"mods={n_models:04d}",
        f"ppm={spread_degree}",
        "fin",
    ]

    model_dir = dirs.MODELS_DIR / config.DATASET.value.name.lower() / "fin"
    train_net_path = utils.construct_filename("-".join(prefix_flds), out_dir=model_dir,
                                              file_ext="pk", add_ds_to_path=False)

    cover_desc = cover_desc.capitalize()
    model_desc = f"{cover_desc}-cover ensemble w. {n_models} models & {spread_degree} spread deg."
    if not train_net_path.exists():
        if is_single:
            learner = SingleCoverEnsemble(n_models=n_models, spread_degree=spread_degree,
                                          opt_params=opt_params)
        else:
            learner = MultiCoverEnsemble(n_models=n_models, spread_degree=spread_degree,
                                         opt_params=opt_params)
        # Single and multicover models are not trained differently. Just differ in
        # how the poison is evaluated
        learner.fit(tg=tg)

        logging.info(f"Saving final {model_desc}...")
        with open(str(train_net_path), "wb+") as f_out:
            pk.dump(learner, f_out)

    # Load the saved module
    logging.info(f"Loading final {model_desc}...")
    with open(str(train_net_path), "rb") as f_in:
        learner = pk.load(f_in)  # CombinedLearner
    return learner
