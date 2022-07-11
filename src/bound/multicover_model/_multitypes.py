__all__ = [
    "MulticoverModel",
]

import abc

from torch import LongTensor, Tensor


class MulticoverModel(abc.ABC):
    r""" Defines the interface for the multicover model interface """
    def calc_coverage(self, y_hat: Tensor, cutoff: Tensor) -> LongTensor:
        r"""
        Given a detailed prediction and cutoff \p cutoff, this function calculates the coverage.
        Approach to calculating coverage is model specific.
        """
        assert y_hat.shape[0] == 1, "Coverage calculated on each instance separately"
        assert cutoff.numel() == 1, "Cutoff should be only a single element"

        coverage = self._calc_coverage(y_hat=y_hat, cutoff=cutoff)
        # Fix the coverage at 1 to align with the standard approach
        coverage.clip_(min=1, max=None)

        assert coverage.numel() == 1, "Only a single coverage element should be present"
        return coverage.view([1, 1])

    @abc.abstractmethod
    def _calc_coverage(self, y_hat: Tensor, cutoff: Tensor) -> LongTensor:
        r"""
        Implements the model specific calculation
        """
