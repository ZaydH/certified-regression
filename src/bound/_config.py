__all__ = [
    "BOUND_DIST",
    "DATASET",
    "DEBUG",
    "DEFAULT_GUROBI_TIMEOUT",
    "GUROBI_TIMEOUT",
    "IS_BOUND_PERCENT",
    "IS_CLASSIFICATION",
    "MIXUP_COPIES",
    "NUM_CLASSES",
    "NUM_TEST_SPLITS",
    "N_DISJOINT_MODELS",
    "IS_NO_MULTICOVER",
    "QUIET",
    "SPREAD_DEGREE",
    "ALT_TYPE",
    "USE_GREEDY",
    "USE_WANDB",
    "VALIDATION_FOLDS",
    "disable_multicover_mode",
    "enable_debug_mode",
    "enable_greedy_mode",
    "is_alt_submodel",
    "override_bound_dist",
    "override_spread_degree",
    "parse",
    "print_configuration",
]

import logging
from pathlib import Path
from typing import Callable, NoReturn, Optional, Union

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt

from .types import AltSubType
from .datasets import SplitDataset

DATASET = None  # type: Optional[SplitDataset]
DATASET_KEY = "dataset"
ALT_TYPE = None  # type: Optional[AltSubType]
TREE_TYPE_KEY = "alt_type"
MODEL_PARAMS = dict()
MODEL_PARAMS_KEY = "model_params"

NUM_CLASSES = None

DEBUG = False

MIXUP_COPIES = None

IS_CLASSIFICATION = False

# Bound Width
BOUND_DIST = None
IS_BOUND_PERCENT = False

DEFAULT_GUROBI_TIMEOUT = 1200
GUROBI_TIMEOUT = DEFAULT_GUROBI_TIMEOUT

N_DISJOINT_MODELS = 51
SPREAD_DEGREE = -1

# Fraction of training samples used for
VALIDATION_FOLDS = 10
NUM_TEST_SPLITS = None

QUIET = False
PLOT = False
USE_WANDB = False

IS_NO_MULTICOVER = False
USE_GREEDY = False

LEARNER_CONFIGS = dict()


def parse(config_file: Union[Path, str]) -> NoReturn:
    r""" Parses the configuration """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Unable to find config file {config_file}")
    if not config_file.is_file():
        raise FileExistsError(f"Configuration {config_file} does not appear to be a file")

    with open(str(config_file), 'r') as f_in:
        all_yaml_docs = YAML().load_all(f_in)

        base_config = next(all_yaml_docs)
        _parse_general_settings(base_config)

    # Sanity checks nothing is out of whack
    _verify_configuration()


def _parse_general_settings(config) -> NoReturn:
    r"""
    Parses general settings for the learning configuration including the dataset, priors, positive
    & negative class information.  It also extracts general learner information
    """
    module_dict = _get_module_dict()
    for key, val in config.items():
        key = key.upper()
        # if key.lower() == CENTROIDS_KEY:
        #     _parse_centroids(module_dict, val)
        # elif key.lower() == DATASET_KEY:
        if key.lower() == DATASET_KEY:
            ds_name = val.upper()
            try:
                module_dict[key] = SplitDataset[ds_name]
            except KeyError:
                raise ValueError(f"Unknown dataset {ds_name}")
        elif key.lower() == TREE_TYPE_KEY:
            tree_type = val.upper()
            module_dict[key] = AltSubType[tree_type]
        elif key.lower() == MODEL_PARAMS_KEY:
            module_dict[key] = _convert_commented_map(val)
        # Drop in replacement field
        else:
            if key not in module_dict:
                raise ValueError(f"Unknown configuration field \"{key}\"")
            module_dict[key] = val

    global BOUND_DIST
    if isinstance(BOUND_DIST, (float, int)):
        BOUND_DIST = [BOUND_DIST]


def _convert_commented_map(com_map: CommentedMap) -> dict:
    r"""
    Ruamel returns variables of type \p CommentedMap. This function converts the \p CommentedMap
    object into a standard dictionary.
    """
    def _convert_val(val):
        if val == "None":
            return None
        if isinstance(val, (float, int, str)):
            return val
        if isinstance(val, ScalarFloat):
            return float(val)
        if isinstance(val, ScalarInt):
            return int(val)
        raise ValueError("Unknown value type in converting CommentedMap")

    return {key: _convert_val(val) for key, val in com_map.items()}


def _get_module_dict() -> dict:
    r""" Standardizes construction of the module dictionary """
    return globals()


def _verify_configuration() -> NoReturn:
    r""" Sanity checks the configuration """
    if DATASET is None:
        raise ValueError("A dataset must be specified")
    # Run one type at a time for simplicity
    # if ALT_TYPE is None:
    #     raise ValueError("An alternate submodel type must be specified")

    pos_params = (
        (N_DISJOINT_MODELS, "Number of disjoint ensemble models"),
        (VALIDATION_FOLDS, "Number of validation folds"),
    )
    for param, name in pos_params:
        if param is not None and param <= 0:
            raise ValueError(f"{name} must be positive")

    _verify_num_models_odd()

    # Verify the bound distance
    if IS_CLASSIFICATION:
        assert BOUND_DIST is None, "Bound distance invalid in classification"
    else:
        assert BOUND_DIST, "Bound distance must be specified"
        # noinspection PyTypeChecker
        assert all(dist > 0 for dist in BOUND_DIST), "All bound distances must be positive"

    if has_gurobi_timeout():
        assert GUROBI_TIMEOUT > 0, "Gurobi timeout must be positive"


def print_configuration(log: Callable = logging.info) -> NoReturn:
    r""" Print the configuration settings """

    def _none_format(_val: Optional[float], format_str: str) -> str:
        if _val is None:
            return "None"
        return f"{_val:{format_str}}"

    log(f"Dataset: {DATASET.name}")

    submodel_str = ALT_TYPE.value if is_alt_submodel() else "NA"
    log(f"Submodel Type: {submodel_str}")
    log(f"Submodel Constructor Parameters: {MODEL_PARAMS}")
    log(f"# Mixup Copies: {MIXUP_COPIES}")
    log(f"# Disjoint Models: {N_DISJOINT_MODELS}")
    log(f"Spreading Degree: {SPREAD_DEGREE}")
    if not IS_CLASSIFICATION:
        log(f"Bound Distance: {BOUND_DIST}")

    log(f"Quiet Mode: {QUIET}")


def reset_learner_settings() -> NoReturn:
    r""" DEBUG ONLY.  Reset the settings specific to individual learners/loss functions """
    global LEARNER_CONFIGS
    LEARNER_CONFIGS = dict()


def set_quiet() -> NoReturn:
    r""" Enables quiet mode """
    global QUIET
    QUIET = True


def enable_debug_mode() -> NoReturn:
    r""" Enables debug mode for the learner """
    global DEBUG
    DEBUG = True


def override_spread_degree(spread_degree: int) -> NoReturn:
    r""" Overrides the training set spread degree """
    global SPREAD_DEGREE
    SPREAD_DEGREE = spread_degree
    assert SPREAD_DEGREE > 0, "Spreading degree must be positive"
    logging.info(f"Overriding spreading degree to \"{SPREAD_DEGREE}\"")


def is_alt_submodel() -> bool:
    r""" Returns \p True if the specified model is an alternate type """
    return ALT_TYPE is not None


def _verify_num_models_odd() -> NoReturn:
    r""" Number of models must be positive and odd """
    assert N_DISJOINT_MODELS > 0, "Number of disjoint models must be positive"
    assert N_DISJOINT_MODELS % 2 == 1, "Number of models must be odd"


def has_gurobi_timeout() -> bool:
    r""" Returns \p True if there is a Gurobi timeout """
    return GUROBI_TIMEOUT is not None


def disable_multicover_mode() -> NoReturn:
    r""" Disables multi-cover mode """
    global IS_NO_MULTICOVER
    IS_NO_MULTICOVER = True


def override_bound_dist(dist: Union[int, float]) -> NoReturn:
    assert dist > 0, "Bound distance must be positive"
    global BOUND_DIST
    logging.debug(f"Overriding bound distance to {BOUND_DIST}")
    BOUND_DIST = [dist]


def enable_greedy_mode() -> NoReturn:
    r""" Disables multi-cover mode """
    global USE_GREEDY
    USE_GREEDY = True
