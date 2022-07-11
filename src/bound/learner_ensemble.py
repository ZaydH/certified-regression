__all__ = [
    "EnsembleLearner",
    "get_group_mask",
    "train_disjoint_ensemble",
]

import abc
import collections
import copy
import dill as pk
import logging
from pathlib import Path
from pickle import UnpicklingError
from typing import List, NoReturn, Optional, Tuple, Union

from sklearn.linear_model import Ridge, RidgeClassifier
from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
from torch import BoolTensor, LongTensor, Tensor
import torch.nn.functional as F  # noqa

from . import _config as config
from . import dirs
from .datasets import utils as ds_utils
from .multicover_model import MulticoverModel
from .multicover_model.knn import MulticoverKnn
from .multicover_model.linear import MulticoverRidge
from .multicover_model.xgb import MulticoverXGB
from .types import BoundReturnType, Coverage, TensorGroup

from . import utils


REG_LOSS = F.l1_loss


class EnsembleLearner(abc.ABC):
    r""" Ensemble regression learner """
    def __init__(self, prefix: str, n_ds_parts: int, model_ds_parts: List[List[int]],
                 opt_params: Optional[dict]):
        super().__init__()
        assert len(model_ds_parts) % 2 == 1, "Number of models expected to be odd"
        assert max(max(parts) for parts in model_ds_parts) < n_ds_parts, "Insufficient ds parts"

        self._prefix = prefix
        self._n_ds_parts = n_ds_parts

        self._model_paths = []

        # Stores the models which each dataset part participated in training
        self._ds_map = None

        self._opt_params = opt_params if opt_params is not None else dict()

        ds_parts_path = utils.construct_filename(prefix=self._prefix + "-ds-parts", file_ext="pk",
                                                 out_dir=self.build_models_dir(), model_num=None,
                                                 add_ds_to_path=False)
        if not ds_parts_path.exists():
            # Stores the dataset parts used by each model
            self._model_ds_parts = model_ds_parts
            with open(ds_parts_path, "wb+") as f_out:
                pk.dump(self._model_ds_parts, f_out)
        else:
            logging.warning(f"Overriding model dataset parts from file \"{ds_parts_path}\"")
            with open(ds_parts_path, "rb") as f_in:
                self._model_ds_parts = pk.load(f_in)

        # Calculate the spread degree
        deg = collections.Counter([part for model_parts in model_ds_parts for part in model_parts])
        least_common, most_common = deg.most_common()[-1], deg.most_common(n=1)[0]
        # Most common returns a tuple of the value and the count. we just want the count
        least_common, most_common = least_common[1], most_common[1]
        assert least_common == most_common, "Inconsistent spread degree"
        self._spread_degree = most_common

        self._tr_size = None

    def name(self) -> str:
        r""" Standardizes name of the ensemble model """
        return self._prefix

    def build_models_dir(self) -> Path:
        r""" Builds the directory to build the models folder """
        return dirs.MODELS_DIR / config.DATASET.value.name.lower() / self._prefix.lower()

    @property
    @abc.abstractmethod
    def cover_type(self) -> str:
        r""" Define the cover type of the model """

    def get_submodel_type(self):
        r""" Gets the class of the submodels """
        assert self._model_paths, "No models to load"
        with open(self._model_paths[0], "rb") as f_in:
            model = pk.load(f_in)
        return model.__class__

    def _log_submodel_size_info(self, model_id: int, sizes: List[int]) -> NoReturn:
        r""" Log information about the submodel dataset sizes """
        logging.info(f"Model {model_id}: Dataset Parts Sizes: {sizes}")
        # Log info about the model
        size = sum(sizes)
        ratio = size / self._tr_size
        logging.info(f"Model {model_id}: Tot. Dataset Size: {size} / {self._tr_size} ({ratio:.2%})")

    def fit(self, tg: TensorGroup):
        r"""
        Fit all models
        :param tg: Training & testing tensors
        :return: Training \p DataLoader
        """
        # Training set size
        self._tr_size = tg.tr_y.numel()

        # self._train_start = time.time()
        for model_id, ds_parts in enumerate(self._model_ds_parts):
            model_path = utils.construct_filename(prefix=self._prefix, file_ext="pk",
                                                  out_dir=self.build_models_dir(),
                                                  model_num=model_id, add_ds_to_path=False)
            self._model_paths.append(model_path)

            if model_path.exists():
                continue

            # Log info about the model
            logging.debug(f"Model ID (out of {self.n_models}): {model_id}")
            logging.info(f"Model {model_id}: Dataset Parts: {sorted(ds_parts)}")

            model = self._train_alt_submodel(tg=tg, model_id=model_id, ds_parts=ds_parts)

            # Logging test error after each model significantly slows down hyperparameter
            # tuning so only log the accuracy/error during normal training.
            if config.IS_CLASSIFICATION:
                self.calc_test_acc(model=model, model_id=model_id, tg=tg)
            elif not config.DATASET.is_weather():
                self.calc_test_err(model=model, model_id=model_id, tg=tg)

            if isinstance(model, nn.Module):
                model.cpu()
            with open(model_path, "wb+") as f_out:
                pk.dump(model, f_out)

    def _train_alt_submodel(self, tg: TensorGroup, model_id: int, ds_parts: List[int]):
        r""" Trains a TabNet learner as the submodel """
        x_tr, y_tr, sizes = _split_train_val(tg=tg, n_ds_parts=self._n_ds_parts,
                                             ds_part_lst=ds_parts)

        self._log_submodel_size_info(model_id=model_id, sizes=sizes)
        # Flatten x since tree assumes the data is a vector
        if len(x_tr.shape) > 2:
            x_tr = x_tr.view([x_tr.shape[0], -1])

        # noinspection PyTypeChecker
        if config.IS_CLASSIFICATION:
            if config.ALT_TYPE.is_multiridge():
                model = MulticoverRidge(**self._opt_params)
            elif config.ALT_TYPE.is_multixgb():
                model = MulticoverXGB(**self._opt_params)
            elif config.ALT_TYPE.is_knn():
                model = MulticoverKnn(**self._opt_params)
            elif config.ALT_TYPE.is_ridge():
                model = RidgeClassifier(**self._opt_params)
            elif config.ALT_TYPE.is_xgb():
                model = XGBClassifier(**self._opt_params)
            else:
                raise ValueError(f"Unknown alternate cls. submodel type {config.ALT_TYPE.name}")
        else:
            if config.ALT_TYPE.is_knn():
                model = MulticoverKnn(**self._opt_params)
            elif config.ALT_TYPE.is_multiridge():
                model = MulticoverRidge(**self._opt_params)
            elif config.ALT_TYPE.is_multixgb():
                model = MulticoverXGB(**self._opt_params)
            elif config.ALT_TYPE.is_ridge():
                model = Ridge(**self._opt_params)
            elif config.ALT_TYPE.is_xgb():
                model = XGBRegressor(**self._opt_params)
            else:
                raise ValueError(f"Unknown alternate reg. submodel type {config.ALT_TYPE.name}")

        # Mark that the model has no test transforms
        model.test_tfms = None
        assert y_tr.numel() > 0, "A model requires at least a single instance"

        with utils.TrainTimer(model_id=model_id, model_name=self.name()):
            if not config.ALT_TYPE.is_multicover_model():
                x_tr, y_tr = utils.build_static_mixed_up(x=x_tr, y=y_tr, include_orig=True)
                model.fit(X=x_tr.numpy(), y=y_tr.numpy())
            else:
                model.fit(X=x_tr.numpy(), y=y_tr.numpy())

        return model

    def calc_test_acc(self, model, model_id: int, tg: TensorGroup) -> NoReturn:
        r""" Log the test accuracy for a single model when performing classification """
        assert config.IS_CLASSIFICATION, "Accuracy only applicable for classification"

        y_hat = self._model_forward(model=model, xs=tg.test_x)
        # noinspection PyUnresolvedReferences
        if len(y_hat.shape) == 2:
            y_hat = y_hat.squeeze(dim=1)
        threshold = self.get_class_threshold()
        y_hat = (y_hat >= threshold).long()

        # Determine fraction correctly predicted
        # noinspection PyTypeChecker
        acc = torch.sum(y_hat == tg.test_y).item() / y_hat.numel()
        logging.info(f"Model {model_id} Test Accuracy: {acc:.2%}")

    def calc_test_err(self, model, model_id: int, tg: TensorGroup) -> NoReturn:
        r"""
        Estimate the test error for the submodel \p model.
        :param model: Submodel of interest
        :param model_id: Identification number of the submodel
        :param tg: Tensor information
        """
        assert not config.IS_CLASSIFICATION, "Test error for direct regression not classification"

        y = tg.test_y
        y_hat = self._model_forward(model=model, xs=tg.test_x)
        # Combine all the results and ensure a valid shape for calculating the loss
        if len(y_hat.shape) == 2:
            y_hat = y_hat.squeeze(dim=1)
        if len(y.shape) == 2:
            y = y.squeeze(dim=1)

        err_vals = (y - y_hat).abs()
        err = REG_LOSS(input=y_hat, target=y)
        logging.info(f"Model {model_id} Mean Test Loss: {err:.4E}")
        percents = [25, 50, 75]
        for per in percents:
            q = torch.tensor([per / 100])
            quant_vals = torch.quantile(err_vals, q=q)
            logging.info(f"Model {model_id}: Quantile {per}% Abs Error: {quant_vals.item():.4E}")

    def forward(self, x: Tensor) -> Tensor:
        r""" Make the prediction across all of the ensemble submodels """
        # noinspection PyUnresolvedReferences
        preds = []
        for model_path in self._model_paths:
            with open(model_path, "rb") as f_in:
                model = pk.load(f_in)

            scores = self._model_forward(model=model, xs=x)

            preds.append(scores)
        # Aggregate across the ensemble
        preds = torch.cat(preds, dim=1)
        return preds

    def forward_wide(self, x: Tensor) -> Tensor:
        r"""
        Special version of the forward method where it uses an underlying \p DataLoader to
        limit the amount each model needs to be reloaded from disk.
        """
        ensemble_preds = []

        for model_path in self._model_paths:
            try:
                model = self._load_submodel(model_path=model_path)
            except EOFError as e:
                raise ValueError(f"Unable to load submodel file \"{model_path}\" with res. {e}")
            model_preds = self._model_forward(model=model, xs=x)

            # All of the example predictions combined
            ensemble_preds.append(model_preds)

        # Aggregate
        ensemble_pred = torch.cat(ensemble_preds, dim=1)
        return ensemble_pred

    def _forward_detail(self, x: Tensor) -> List[List[Tensor]]:
        r"""
        Special version of the forward method where it uses an underlying \p DataLoader to
        limit the amount each model needs to be reloaded from disk.
        """
        ex_preds = [[] for _ in range(x.shape[0])]

        for model_path in self._model_paths:
            with open(model_path, "rb") as f_in:
                model = pk.load(f_in)
            ys = model.predict_detail(x)
            for idx in range(x.shape[0]):
                ex_preds[idx].append(ys[idx].view([1, -1]))

        assert len(ex_preds) == x.shape[0], "Mismatch in example counts"
        assert all(len(vals) == self.n_models for vals in ex_preds), "Mismatch with model count"
        return ex_preds

    @staticmethod
    def _load_submodel(model_path: Path):
        r""" Standardizes loading a submodel path """
        try:
            assert model_path.exists(), f"Load submodule \"{model_path}\" but file does not exist"
            with open(model_path, "rb") as f_in:
                return pk.load(f_in)
        except UnpicklingError as e:
            logging.error(f"Error opening file {model_path}")
            raise e

    @staticmethod
    def _model_forward(model, xs: Tensor) -> Tensor:
        r""" Standardizes the forward method for submodels as it can differ by setup """
        if isinstance(model, nn.Module):
            return model.forward(xs)

        xs = xs.view([xs.shape[0], -1]).cpu().numpy()
        # Custom for models using the sklearn API
        y_hat = model.predict(xs)
        y_hat = torch.from_numpy(y_hat).float().view([-1, 1])
        return y_hat

    @property
    def n_models(self) -> int:
        r""" Number of submodels used in the ensemble """
        return len(self._model_ds_parts)

    def is_disjoint(self) -> bool:
        r""" Default is that the model does not train on disjoint sets """
        return self._spread_degree == 1

    @staticmethod
    def is_multicover() -> bool:
        r""" Return \p True if the model supports multi-coverage """
        return False

    @property
    def n_ds_parts(self) -> int:
        r""" Number of dataset parts used by the model """
        return self._n_ds_parts

    @property
    def ppm(self) -> int:
        r""" Number of dataset parts per submodel """
        return self._spread_degree

    def get_submodel_ds_parts(self, model_id: int) -> List[int]:
        r""" Accessor for the dataset parts used to train the submodel """
        assert 0 <= model_id < self.n_models, f"Model ID {model_id} not in [0,{self.n_models})"
        return copy.deepcopy(self._model_ds_parts[model_id])

    def calc_prediction(self, ys: Tensor) -> Tensor:
        r""" Calculate the predicted output """
        assert ys.shape[1] == self.n_models, "Mismatch between number of models and the dimension"
        vals, _ = ys.median(dim=1)
        return vals

    @staticmethod
    def get_class_threshold() -> float:
        r""" Returns the classification threshold for distinguishing the two classes """
        if config.is_alt_submodel():
            return (ds_utils.NEG_LABEL + ds_utils.POS_LABEL) / 2
        return 0

    def _get_ds_map(self) -> List[List[int]]:
        r"""
        Constructs the mapping of dataset parts to models.  This is primarily used when each
        model is not trained on disjoint dataset.
        """
        if self._ds_map is not None:
            return self._ds_map

        self._ds_map = [[] for _ in range(self._n_ds_parts)]
        # Iterate through all the models and add that model ID number to the corresponding
        # dataset part location
        for model_id, ds_parts in enumerate(self._model_ds_parts):
            # assert len(ds_parts) == len(set(ds_parts)), "Duplicate dataset parts observed"
            for part_id in ds_parts:
                self._ds_map[part_id].append(model_id)

        return self._ds_map

    def _med_model_cnt(self) -> int:
        r""" Model ID number for the median prediction """
        return self.n_models // 2

    def _calc_bound_dists(self, full_yhat: Tensor,
                          lbound_change: Optional[Union[float, Tensor]] = None,
                          ubound_change: Optional[Union[float, Tensor]] = None) \
            -> Tuple[Optional[LongTensor], Optional[LongTensor]]:
        r"""
        Calculates the perturbation distance up and down to remain within the range
        median +/- \p pred_chg.

        :param full_yhat: Model prediction values
        :param lbound_change: Amount to change the lower bound. If None, ignore the lower bound
        :param ubound_change: Amount to change the upper bound. If None, ignore the upper bound

        :return: Tuple of the lower and upper perturbation distances respectively
        """
        assert lbound_change is not None or ubound_change is not None, "No change specified"

        # half_pt = self._med_model_cnt()
        # Update to a more general search routine where the distance code can also find
        # coverage values
        half_pt = full_yhat.shape[1] // 2

        # Sort along the prediction axis to calculate the bound
        full_yhat, _ = full_yhat.sort(dim=1)
        yhat_med, _ = full_yhat.median(dim=1)

        # ToDo consider restore
        if full_yhat.shape[1] & 1 == 1:
            # For odd model counts, verify median since exact
            mid_yhat = full_yhat[:, half_pt]
            assert torch.all(torch.eq(yhat_med, mid_yhat)), "Unexpected median calculation"

        # ToDo decide what to do when the number of examples is even

        def _check_change_var(change_var: Union[float, Tensor], mask: Union[bool, BoolTensor]):
            r""" Verify the input change variable matches the expected sign """
            cond = (isinstance(change_var, (int, float)) and mask) or \
                   (isinstance(change_var, Tensor) and torch.all(mask))
            assert cond, "Invalid change variable"

        lbound = ubound = None
        if lbound_change is not None:
            _check_change_var(lbound_change, mask=lbound_change <= 0)
            lb_vals = yhat_med + lbound_change  # type: Tensor

            # Negate so the minimum becomes the maximum
            # Flip is needed to since searchsorted assumes monotonically increasing
            tmp_full, lb_vals = full_yhat.flip(dims=[1]).neg(), lb_vals.unsqueeze(dim=1).neg()
            # Need to subtract one as searchsorted returns location above the threshold
            lbound = torch.searchsorted(tmp_full, lb_vals) - 1

            self._check_bound(bound=lbound, full_vals=tmp_full, max_vals=lb_vals)
            lbound -= half_pt

        if ubound_change is not None:
            _check_change_var(ubound_change, mask=ubound_change >= 0)
            ub_vals = (yhat_med + ubound_change).unsqueeze(dim=1)
            # Need to subtract one as searchsorted returns location above the threshold
            ubound = torch.searchsorted(full_yhat, ub_vals) - 1

            self._check_bound(bound=ubound, full_vals=full_yhat, max_vals=ub_vals)
            ubound -= half_pt

        # noinspection PyTypeChecker
        return lbound, ubound

    def _check_bound(self, bound: Tensor, full_vals: Tensor, max_vals: Tensor) -> NoReturn:
        r""" Verify that the bound is actually a correct transition """
        n_model = full_vals.shape[1]
        # Prevent dimension explosions when using these vectors to index
        bound, max_vals = bound.squeeze(), max_vals.squeeze(dim=1)

        def _get_vals(idx: LongTensor) -> Tensor:
            vals = full_vals[torch.arange(full_vals.shape[0]), idx]
            return vals

        # Below the threshold.  Search sorted returns the first index LESS THAN (but NOT equal to)
        # the specified threshold.
        mask = _get_vals(bound) < max_vals  # type: BoolTensor
        # In some cases, the bound may be less than all the predictions. THis checks that condition
        all_gt = full_vals[:, 0] <= max_vals
        # If all values are equal to the max_value, then the above max_vals check can fail.
        # This corner check is added to handle such cases. This case is primarily seen in MNIST
        # classification.
        all_eq = torch.all(full_vals == max_vals.unsqueeze(dim=1), dim=1)
        # In some cases, you can have nan medians when the median prediction is -inf or inf.
        # Consider that case.
        is_nan = max_vals.isnan()

        # noinspection PyArgumentList
        mask = mask.logical_or(all_eq).logical_or(is_nan).logical_or(all_gt)
        assert torch.all(mask), "Bound not less than max"

        # Above the threshold
        bound = bound + 1
        # Need to clamp bound to ensure it does not exceed length of tensor
        tmp_bound = bound.clamp_max(n_model - 1)
        mask = (bound == n_model).logical_or(_get_vals(tmp_bound) >= max_vals)
        assert torch.all(mask), "Bound not less than max"

    def calc_bound(self, full_yhat: Tensor, ids: LongTensor,
                   l_cutoff_dist: Optional[Union[float, Tensor]] = None,
                   u_cutoff_dist: Optional[Union[float, Tensor]] = None,
                   coverage: Optional[LongTensor] = None) -> BoundReturnType:
        r""" Calculates the regression perturbation bound """
        assert len(full_yhat.shape) == 2, "full_yhat should have two arguments"
        assert full_yhat.shape[0] == ids.numel(), "Mismatch in count"
        assert u_cutoff_dist is None or u_cutoff_dist.shape[0] == full_yhat.shape[0], \
            "Up. bound count"
        assert l_cutoff_dist is None or l_cutoff_dist.shape[0] == full_yhat.shape[0], \
            "Low bound count"

        if full_yhat.shape[1] == 1 and not self.is_multicover():
            logging.warning(f"Bound calculation for single element. Skipping")
            bound = torch.zeros_like(ids, dtype=torch.long)
            grb_info = None
        else:
            bound, grb_info = self._calc_bound(full_yhat=full_yhat, ids=ids,
                                               u_cutoff_dist=u_cutoff_dist,
                                               l_cutoff_dist=l_cutoff_dist,
                                               coverage=coverage)
        # Make the bound distances a vector
        if len(bound.shape) > 1:
            bound = bound.squeeze(dim=1)
        assert len(bound.shape) == 1, "Bound tensor should be 1D"
        return bound, grb_info

    @abc.abstractmethod
    def _calc_bound(self, full_yhat: Tensor, ids: LongTensor,
                    l_cutoff_dist: Optional[Union[float, Tensor]],
                    u_cutoff_dist: Optional[Union[float, Tensor]],
                    coverage: Optional[LongTensor]) -> BoundReturnType:
        r""" Calculates the regression perturbation bound """

    def calc_coverage(self, x: Tensor, lbound: Optional[Tensor],
                      ubound: Optional[Tensor]) -> Coverage:
        r""" Calculate the coverage of each model for each training example and each model """
        assert self.is_multicover(), "Coverage only applicable for multicover models"
        assert lbound is not None or ubound is not None, "No bound specified to calculate coverage"

        msg = "Coverage forward detail"
        logging.debug(f"Starting: {msg}...")
        y_pred = self._forward_detail(x=x)
        logging.debug(f"COMPLETED: {msg}")

        cover = [[] for _ in range(x.shape[0])]
        l_cover = copy.deepcopy(cover) if lbound is not None else None
        u_cover = copy.deepcopy(cover) if ubound is not None else None
        for i_model, submodel_path in enumerate(self._model_paths):
            submodel = self._load_submodel(model_path=submodel_path)

            assert isinstance(submodel, MulticoverModel), "Submodel is not a multicover model"

            for i_ele in range(x.shape[0]):
                lbound_i = lbound[i_ele:i_ele + 1] if lbound is not None else None
                ubound_i = ubound[i_ele:i_ele + 1] if ubound is not None else None

                full_yhat = y_pred[i_ele][i_model]
                if config.DATASET.is_expm1_scale():
                    full_yhat = ds_utils.scale_expm1_y(y=full_yhat)

                if lbound_i is not None:
                    # Negate the values since the coverage function checks for upper bounds
                    cover = submodel.calc_coverage(y_hat=-full_yhat, cutoff=-lbound_i)
                    l_cover[i_ele].append(cover)

                if ubound_i is not None:
                    cover = submodel.calc_coverage(y_hat=full_yhat, cutoff=ubound_i)
                    u_cover[i_ele].append(cover)

        return Coverage(l_cover=_flatten_cover(l_cover), u_cover=_flatten_cover(u_cover))


def _flatten_cover(_cover: Optional[List[List[LongTensor]]]) -> Optional[LongTensor]:
    if _cover is None:
        return _cover
    _cover = [torch.cat(submodel_cover, dim=1) for submodel_cover in _cover]
    # noinspection PyTypeChecker
    _cover = torch.cat(_cover, dim=0)
    # noinspection PyArgumentList
    _cover.clip_(min=1)
    # noinspection PyTypeChecker
    return _cover


class DisjointEnsemble(EnsembleLearner):
    r""" Special version of ensemble where the dataset parts are disjoint """
    def __init__(self, n_models: int, prefix: str = "", opt_params: Optional[dict] = None):
        model_ds_parts = [[i] for i in range(n_models)]

        flds = []
        if config.is_alt_submodel():
            flds.append(config.ALT_TYPE.name.lower())
        if prefix:
            flds.append(prefix)
        flds += [self.cover_type, f"{n_models:04d}"]

        super().__init__(prefix="-".join(flds), n_ds_parts=n_models,
                         model_ds_parts=model_ds_parts, opt_params=opt_params)

    def _calc_bound(self, full_yhat: Tensor, ids: LongTensor,
                    l_cutoff_dist: Optional[Union[float, Tensor]],
                    u_cutoff_dist: Optional[Union[float, Tensor]],
                    coverage: Optional[LongTensor]) -> BoundReturnType:
        r""" Calculates the regression perturbation bound """
        if coverage is not None:
            raise NotImplementedError("Coverage not yet supported for disjoint models")

        assert not config.USE_GREEDY, "Greedy not applicable for disjoint ensembles"
        assert full_yhat.shape[1] == self.n_models, "Mismatch between num results and num models"

        assert l_cutoff_dist is not None or u_cutoff_dist is not None, "No bound dist. specified"

        ldist, udist = self._calc_bound_dists(full_yhat=full_yhat, lbound_change=l_cutoff_dist,
                                              ubound_change=u_cutoff_dist)

        bound = ldist if (ldist is not None) else udist
        if udist is not None:
            # Bound is the minimum of an increasing or decreasing perturbation
            # noinspection PyTypeChecker
            bound = torch.min(bound, udist)

        # noinspection PyTypeChecker
        return bound, None

    @property
    def cover_type(self) -> str:
        r""" Define the cover type of the model """
        return "disjoint"


def _split_train_val(tg: TensorGroup, ds_part_lst: List[int],
                     n_ds_parts: int) -> Tuple[Tensor, Tensor, List[int]]:
    r""" Groups X and y into train and validation based on the dataset parts """
    # Partition randomly into groups
    group_id = tg.tr_hash % n_ds_parts  # type: LongTensor

    # Since some part numbers may be duplicated for the same model, need to iterate
    # through each part number separately and iteratively build the dataset rather
    # than using a single mask as the original implementation did.
    x_lst, y_lst, val_mask_lst, sizes = [], [], [], []
    for part_id in ds_part_lst:
        part_lst = [part_id]
        grp_mask = get_group_mask(group_id=group_id, group_lst=part_lst)
        sizes.append(torch.sum(grp_mask).item())

        x_lst.append(tg.tr_x[grp_mask])
        y_lst.append(tg.tr_y[grp_mask])
        #
        # # Partition into train and validation
        # val_mask = get_validation_mask(hash_vals=tg.tr_hash[grp_mask], n_ds_parts=n_ds_parts)
        # val_mask_lst.append(val_mask)

    return torch.cat(x_lst, dim=0), torch.cat(y_lst, dim=0), sizes


def get_group_mask(group_id: LongTensor, group_lst: List[int]) -> BoolTensor:
    r""" Returns a mask whether each training example's group ID is in the group ID list """
    group_lst = torch.tensor(group_lst, dtype=torch.long)
    mask = in1d(group_id, group_lst)
    assert mask.numel() == group_id.numel() and mask.shape[0] == group_id.numel(), \
        "Unexpected shape for the mask"
    return mask


# def get_validation_mask(hash_vals: LongTensor, n_ds_parts: int) -> BoolTensor:
#     r""" Returns \p True for each element in \p hash_vals if that value is used for validation """
#     if config.DATASET.is_housing():
#         # noinspection PyTypeChecker
#         return torch.zeros_like(hash_vals, dtype=torch.bool)
#
#     val_fold_ids = (hash_vals // n_ds_parts) % config.VALIDATION_FOLDS
#     is_valid = val_fold_ids == 0
#     return is_valid


def in1d(ar1, ar2) -> BoolTensor:
    r""" Returns \p True if each element in \p ar1 is in \p ar2 """
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def train_disjoint_ensemble(tg: TensorGroup,
                            opt_params: Optional[dict] = None) -> DisjointEnsemble:
    r"""
    Train an ensemble where each submodel is trained on a disjoint dataset

    :param tg: \p TensorGroup
    :param opt_params: Optional model parameters. Primarily used for hyperparameter tuning
    :return: Collection of trained classifiers
    """
    n_models = config.N_DISJOINT_MODELS

    # Prefix used for the serialized backup
    prefix_flds = []
    if config.is_alt_submodel():
        prefix_flds.append(config.ALT_TYPE.name.lower())
    prefix_flds += [
        "disjoint",
        f"{n_models:04d}",
        "fin",
    ]

    model_dir = dirs.MODELS_DIR / config.DATASET.value.name.lower() / "fin"
    train_net_path = utils.construct_filename("-".join(prefix_flds), out_dir=model_dir,
                                              file_ext="pk", add_ds_to_path=False)

    # Model description only used for logging serialization info
    model_str = f"{n_models}"
    if config.is_alt_submodel():
        model_str = f"{model_str} {config.ALT_TYPE.value}"
    model_desc = f"Disjoint ensemble with {model_str} models"

    if not train_net_path.exists():
        learner = DisjointEnsemble(n_models=n_models, opt_params=opt_params)
        learner.fit(tg=tg)

        logging.info(f"Saving final {model_desc}...")
        with open(str(train_net_path), "wb+") as f_out:
            pk.dump(learner, f_out)

    # Load the saved module
    logging.info(f"Loading final {model_desc}...")
    with open(str(train_net_path), "rb") as f_in:
        learner = pk.load(f_in)  # CombinedLearner
    return learner
