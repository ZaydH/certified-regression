__all__ = [
    "BoundReturnType",
    "Coverage",
    "GurobiResList",
    "TensorGroup",
    "AltSubType",
]

import dataclasses
import enum
import hashlib
from typing import List, Optional, NoReturn, Tuple

import numpy as np

import torch
from torch import LongTensor, Tensor

from .gurobi_utils import GurobiResult

# Type returned by the Gurobi result parser
GurobiResList = List[Tuple[Optional[GurobiResult], Optional[GurobiResult]]]
BoundReturnType = Tuple[LongTensor, Optional[GurobiResList]]


class AltSubType(enum.Enum):
    KNN = "KNN"
    MULTIRIDGE = "MultiRidge"
    MULTIXGB = "MultiXGBoost"
    RIDGE = "Ridge"
    XGB = "XGBoost"

    def is_knn(self) -> bool:
        return self == self.KNN

    def is_multiridge(self) -> bool:
        return self == self.MULTIRIDGE

    def is_multixgb(self) -> bool:
        return self == self.MULTIXGB

    def is_ridge(self) -> bool:
        r""" Return \p True if the submodel is simple ridge regression """
        return self == self.RIDGE

    def is_xgb(self) -> bool:
        r""" Return \p True if the submodel is simple XGBoost """
        return self == self.XGB

    def is_multicover_model(self) -> bool:
        r""" Returns \p True if the submodel supports multiple cover """
        vals = [
            self.is_knn(),
            self.is_multiridge(),
            self.is_multixgb(),
        ]
        return any(vals)


@dataclasses.dataclass
class TensorGroup:
    r""" Encapsulates a group of tensors used by the learner """
    tr_x: Tensor = torch.zeros(0, dtype=torch.float)
    tr_y: LongTensor = torch.zeros(0, dtype=torch.long)
    tr_lbls: LongTensor = torch.zeros(0, dtype=torch.long)
    tr_ids: LongTensor = torch.zeros(0, dtype=torch.long)
    tr_hash: LongTensor = torch.zeros(0, dtype=torch.long)

    test_x: Tensor = torch.zeros(0, dtype=torch.float)
    test_y: LongTensor = torch.zeros(0, dtype=torch.long)
    test_lbls: LongTensor = torch.zeros(0, dtype=torch.long)
    test_ids: LongTensor = torch.zeros(0, dtype=torch.long)

    def __len__(self) -> int:
        r""" Gets the length of the \p TensorGroup """
        tr_len = len(self.tr_x)
        assert tr_len == len(self.tr_y) == len(self.tr_ids), "Training length mismatch"
        return tr_len

    def calc_tr_hash(self) -> NoReturn:
        r""" Construct training set hash """
        assert self.tr_y.numel() > 0, "Training set appears empty"
        assert self.tr_y.numel() == self.tr_x.shape[0] == self.tr_lbls.numel(), \
            "Training element count mismatch"

        dtype = self.tr_x.dtype
        # Numpy expects a flat x vector
        tr_x = self.tr_x.view([self.tr_y.numel(), -1])
        tr_y, tr_lbls = self.tr_y.view([-1, 1]).type(dtype), self.tr_lbls.view([-1, 1]).type(dtype)
        # rand_vals ensures different partitioning across different iterations
        rand_vals = torch.rand(tr_y.shape, dtype=dtype)
        # Combine all elements into one tensor
        all_tr_data = torch.cat([tr_x, tr_y, tr_lbls, rand_vals], dim=1)

        self.tr_hash = calc_hash_tensor(np_arr=all_tr_data.numpy())

    def build_ids(self) -> NoReturn:
        r""" Each training and test instance is assigned a unique ID number """
        self.tr_ids = torch.arange(self.tr_y.numel(), dtype=torch.long).long()

        self.test_ids = self.tr_ids.numel() + torch.arange(self.test_y.numel(), dtype=torch.long)


def calc_hash_tensor(np_arr: np.ndarray) -> LongTensor:
    r""" Hashes a numpy array along the first dimension of \p np_arr """
    hash_vals = []
    for i in range(np_arr.shape[0]):
        str_val = str(np_arr[i].data.tobytes())
        hash_obj = hashlib.sha256(str_val.encode())

        digest = hash_obj.hexdigest()
        # Torch longs are only 8 bytes so restrict digest to a maximum of 8 bytes
        digest = digest[-min(8, len(digest)):]
        # Convert to an integer.  Need to specify the number as base 16
        digest_int = int(digest, 16)

        hash_vals.append(digest_int)

    hash_tensor = torch.tensor(hash_vals, dtype=torch.long).long()
    return hash_tensor


@dataclasses.dataclass
class Coverage:
    """ Coverage results for multicover """
    l_cover: Optional[LongTensor]
    u_cover: Optional[LongTensor]
