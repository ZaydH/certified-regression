__all__ = [
    "MulticoverXGB",
]

from xgboost import XGBClassifier, XGBRegressor

from .. import _config as config
from .linear import BaseMultiLinear

RANDOM_SEED = 42


class FixedXGB(XGBRegressor):
    def __init__(self, **kwargs):
        kwargs["random_state"] = RANDOM_SEED


class MulticoverXGB(BaseMultiLinear):
    r""" Same as multicover """

    def get_base_learner_class(self, **kwargs):
        r""" Accesses the base class of the learner """
        base_cls = XGBClassifier if config.IS_CLASSIFICATION else XGBRegressor
        return base_cls(**kwargs)
