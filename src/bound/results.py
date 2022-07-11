__all__ = [
    "calc",
]

import logging
from typing import NoReturn

import gurobipy as grb

from . import _config as config
from .learner_ensemble import EnsembleLearner
from . import results_utils
from .types import TensorGroup


def calc(model: EnsembleLearner, tg: TensorGroup) -> NoReturn:
    if config.IS_CLASSIFICATION:
        return results_utils.classification.calc(model=model, tg=tg)
    elif config.DATASET.is_tabular():
        return results_utils.regression.calc(model=model, tg=tg)
    else:
        raise ValueError("Unknown how to calculate results for this experiment type")
