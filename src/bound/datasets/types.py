__all__ = [
    "SplitDataset",
    "ViewModule",
    "ViewTo1D",
]

import collections
from enum import Enum
from typing import List

from torch import Tensor
import torch.nn as nn

DatasetParams = collections.namedtuple("DatasetParams", "name dim n_classes")
CIFAR_DIM = [3, 32, 32]
MNIST_DIM = [1, 28, 28]
WEATHER_DIM = []


# noinspection PyPep8Naming
class SplitDataset(Enum):
    r""" Valid datasets for testing """
    AMES_HOUSING = DatasetParams("Ames-Housing", [359], -1)
    AUSTIN_HOUSING = DatasetParams("Austin-Housing", [345], -1)

    DIAMONDS = DatasetParams("Diamonds", [26], -1)

    LIFE = DatasetParams("Life", [], -1)

    SPAMBASE = DatasetParams("Spambase", [57], 2)

    WEATHER = DatasetParams("Weather", WEATHER_DIM, -1)

    def is_spambase(self) -> bool:
        r""" Returns \p True if the Adult dataset """
        return self == self.SPAMBASE

    def is_diamonds(self) -> bool:
        r"""
        Returns \p True if the dataset is the Kaggle diamonds dataset. See:
        https://www.kaggle.com/datasets/shivam2503/diamonds
        """
        return self == self.DIAMONDS

    def is_life(self) -> bool:
        r""" Returns \p True if the Diabetes dataset """
        return self == self.LIFE

    def is_weather(self) -> bool:
        r""" Returns \p True if the dataset is the Shifts-Weather dataset """
        return self == self.WEATHER

    def is_austin_housing(self) -> bool:
        r""" Returns \p True if the dataset is the Austin Housing one """
        return self == self.AUSTIN_HOUSING

    def is_housing(self) -> bool:
        r""" Returns \p True if the dataset is either the Ames or Austin housing dataset """
        return self in (self.AMES_HOUSING, self.AUSTIN_HOUSING)

    def is_tabular(self) -> bool:
        r""" Return \p True if using a tabular dataset """
        vals = [
            self.is_diamonds(),
            self.is_housing(),
            self.is_life(),
            self.is_spambase(),
            self.is_weather(),
        ]
        return any(vals)

    def is_expm1_scale(self) -> bool:
        r""" Returns \p True if the dataset requires scaling y by expm1 """
        return self.is_diamonds() or self.is_housing()

    def has_neg_label(self) -> bool:
        r"""
        Returns \p if the dataset contains negative labels.
        """
        return self.is_weather()


class ViewModule(nn.Module):
    r""" General view layer to flatten to any output dimension """
    def __init__(self, d_out: List[int]):
        super().__init__()
        self._d_out = tuple(d_out)

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        # noinspection PyUnresolvedReferences
        return x.reshape((x.shape[0], *self._d_out))


class ViewTo1D(ViewModule):
    r""" View layer simplifying to specifically a single dimension """
    def __init__(self):
        super().__init__([-1])
