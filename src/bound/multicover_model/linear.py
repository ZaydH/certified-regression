__all__ = [
    "BaseMultiLinear",
    "MulticoverRidge",
]

import abc
import copy
from typing import List, Union

import numpy as np
from sklearn.linear_model import  Ridge, RidgeClassifier

import torch
from torch import LongTensor, Tensor

from .. import _config as config
if __name__ != "tree":
    from . import _multitypes
else:
    import _multitypes
from .. import utils as parent_utils


class BaseMultiLinear(_multitypes.MulticoverModel):
    _BASE_TFMS = None

    def __init__(self, **kwargs):
        super().__init__()
        self._all_tfms = None if self._BASE_TFMS is None else self._BASE_TFMS()
        self._all_data = self.get_base_learner_class(**kwargs)

        self._sub_models = None
        self._sub_tfms = None

    def fit(self, X, y, **kwargs):
        r""" Fit the learner to the training data """
        assert y.shape[0] > 1, "Two training instances needed to fit multicover models"

        # XGB Classifier requires label 0 before first so just sort the instances
        if config.IS_CLASSIFICATION:
            def _make_tensor(val: Union[np.ndarray, Tensor]) -> Tensor:
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                return val.cpu()
            X, y = _make_tensor(X), _make_tensor(y)
            y, sort_idx = torch.sort(y)
            X = X[sort_idx]

        if isinstance(X, Tensor):
            X = X.numpy()

        n_ele = y.shape[0]
        # LOO training
        self._sub_models = [copy.deepcopy(self._all_data) for _ in range(n_ele)]
        if self._all_tfms is not None:
            self._sub_tfms = [copy.deepcopy(self._all_tfms) for _ in range(n_ele)]

        # Model trained on all the data
        x_tr, y_tr = parent_utils.build_static_mixed_up(x=X, y=y, include_orig=True)
        if self._all_tfms is not None:
            self._all_tfms.fit(x_tr)
            x_tr = self._all_tfms.transform(x_tr)
        self._all_data.fit(x_tr, y_tr, **kwargs)

        # Perform leave one out training of each model to construct a deletion bound
        ele_range = torch.arange(n_ele)
        X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        for i in range(n_ele):
            mask = ele_range != i
            x_sub, y_sub = X[mask], y[mask]
            x_sub, y_sub = parent_utils.build_static_mixed_up(x=x_sub, y=y_sub,
                                                              include_orig=True)

            # Transform the X if specified
            if self._sub_tfms is not None:
                tfms = self._sub_tfms[i]
                tfms.fit(x_sub)
                x_sub = tfms.transform(x_sub)

            self._sub_models[i].fit(x_sub, y_sub.numpy())

    def predict(self, X) -> np.ndarray:
        r""" Predictions performed using the best model """
        if isinstance(X, Tensor):
            X = X.numpy()
        if self._all_tfms is not None:
            X = self._all_tfms.transform(X)
        return self._all_data.predict(X)

    def predict_detail(self, x: Tensor) -> List[Tensor]:
        r""" Returns detailed predictions of y values for each training example """
        all_pred, x = [], x.numpy()
        # Perform a prediction with each submodel
        for i, submodel in enumerate(self._sub_models):
            if self._sub_tfms is not None:
                tfms = self._sub_tfms[i]
                x = tfms.transform(x)

            yhat = submodel.predict(x)
            yhat = torch.from_numpy(yhat).view([x.shape[0], -1])
            all_pred.append(yhat)
        # Combine all the predictions into one tensor
        all_pred = torch.cat(all_pred, dim=1)

        pred_detail = [all_pred[i:i + 1] for i in range(x.shape[0])]
        return pred_detail

    def _calc_coverage(self, y_hat: Tensor, cutoff: Tensor) -> LongTensor:
        r""" Calculates coverage specific to trees """
        base_coverage = torch.ones_like(cutoff, dtype=torch.long)

        y_hat_max, _ = torch.max(y_hat, dim=1)
        diff = y_hat_max <= cutoff
        coverage = base_coverage + diff
        return coverage

    def n_sub(self) -> int:
        return len(self._sub_models)

    @abc.abstractmethod
    def get_base_learner_class(self, **kwargs):
        r""" Accesses the base class of the learner """


class MulticoverRidge(BaseMultiLinear):
    def get_base_learner_class(self, **kwargs):
        r""" Accesses the base class of the learner """
        base_cls = RidgeClassifier if config.IS_CLASSIFICATION else Ridge
        return base_cls(**kwargs)
