__all__ = [
    "MulticoverKnn",
]

from typing import List

from sklearn.neighbors import KNeighborsRegressor

import torch
from torch import LongTensor, Tensor

if __name__ != "knn":
    from . import _multitypes
else:
    import _multitypes


class MulticoverKnn(_multitypes.MulticoverModel, KNeighborsRegressor):
    # def __init__(self, **kwargs):
    #     super(types.MulticoverModel).__init__()
    #     super(KNeighborsRegressor).__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        # setattr(self, "y", y if isinstance(y, Tensor) else torch.from_numpy(y))
        n_x = X.shape[0]
        if n_x & 1 == 0:
            n_x -= 1
        self.n_neighbors = min(self.n_neighbors, n_x)

        if isinstance(X, Tensor):
            X = X.cpu().numpy()

        # noinspection PyUnresolvedReferences,PyArgumentList
        super().fit(X, y, **kwargs)

    def predict(self, X):
        if isinstance(X, Tensor):
            X = X.numpy()
        vals = self.kneighbors(X, return_distance=False)
        y = torch.from_numpy(self._y)
        y_vals = y[vals]
        y_vals, _ = y_vals.sort(dim=1)
        k = self.n_neighbors
        return y_vals[:, self.n_neighbors // 2].numpy()

    def predict_detail(self, x: Tensor) -> List[Tensor]:
        r""" Returns detailed predictions of y values for each training example """
        vals = [self.kneighbors(x[i:i+1].numpy(), return_distance=False) for i in range(x.shape[0])]
        vals = [torch.from_numpy(val) for val in vals]
        y = torch.from_numpy(self._y)
        vals = [y[val] for val in vals]
        return vals

    def _calc_coverage(self, y_hat: Tensor, cutoff: Tensor) -> LongTensor:
        r""" Calculates coverage specific to trees """
        return _calc_coverage(y_hat=y_hat, cutoff=cutoff)


def _calc_coverage(y_hat: Tensor, cutoff: Tensor) -> LongTensor:
    r"""
    Calculates coverage specific to random-split trees

    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([3.5])).item()
    2
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([3.4])).item()
    2
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([5])).item()
    3
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([2.6])).item()
    1
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([2.3])).item()
    1
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([1.6])).item()
    -1
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([1.3])).item()
    -1
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([0.3])).item()
    -2
    >>> _calc_coverage(torch.tensor([list(range(5))]), cutoff=torch.tensor([-0.3])).item()
    -3
    """
    assert y_hat.shape[1] & 1 == 1, "Odd k expected"
    assert y_hat.shape[1] > 1, "A k larger than 1 is required for multicover to make sense"

    # y_hat, _ = y_hat.sort(dim=1)
    if len(cutoff.shape) == 1:
        cutoff = cutoff.unsqueeze(dim=1)
    # assert len(y_hat.shape) == 1, "Only a single dimension is expected"
    # cut_loc = torch.searchsorted(y_hat, cutoff)
    coverage = (y_hat < cutoff.item()).sum(dim=1) - y_hat.shape[1] // 2 + 1
    # Negative coverage if below the midpoint
    coverage[coverage <= 0] -= 1

    return coverage
