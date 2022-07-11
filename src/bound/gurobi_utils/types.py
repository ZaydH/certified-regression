__all__ = [
    "GurobiNoLicenseKeyError",
    "GurobiRequestFailedError",
    "GurobiResult",
    "GurobiStatusCode",
]

import dataclasses
import enum
import logging
from typing import NoReturn, Set, Union

import numpy as np

import gurobipy as grb
from gurobipy import GRB

from .. import _config as config


class GurobiNoLicenseKeyError(Exception):
    r""" Raised when there are no keys left in the key source file """
    pass


class GurobiRequestFailedError(Exception):
    r""" Raised when the request for the key failed """
    pass


@dataclasses.dataclass
class GurobiResult:
    r""" Stores the results of the Gurobi solver """
    ex_id: int
    is_relax: bool
    is_lower: bool
    is_single: bool

    obj_val: Union[float, int]
    low_bound: Union[float, int]

    runtime: float
    timed_out: bool

    @classmethod
    def create(cls, ilp: grb.Model, ex_id: int, is_lower: bool, relax: bool,
               is_single: bool) -> "GurobiResult":
        r""" Factory method to construct the Gurobi result """
        status_code = GurobiStatusCode(ilp.Status)
        # Check if the status code reports the model timed out
        user_bd = status_code.USER_OBJ_LIMIT
        timed_out = status_code == GurobiStatusCode.TIME_LIMIT
        assert status_code in GurobiStatusCode.get_valid(), \
            f"Unexpected Gurobi status {status_code}"

        # Check on the runtime and make sure it is valid
        runtime = ilp.Runtime
        assert 0 < runtime, "A positive runtime is expected"

        low_bound = ilp.getAttr(GRB.Attr.ObjBound)  # Lower bound on objective value
        obj_value = ilp.getObjective().getValue()  # Best feasible solution found
        if not relax:
            def _to_int(_x) -> int:
                r""" Prevent weird rounding errors """
                return int(np.floor(_x + 0.5))

            # Use round since small floating point discrepancies can cause the wrong number
            # to be returned
            low_bound, obj_value = _to_int(low_bound), _to_int(obj_value)

        # Solver stops when the optimal solution is with the MIP tolerance gap (default 1e-4).
        # Check either an "optimal" solution or timeout.
        if obj_value > 0:
            gap = (obj_value - low_bound) / obj_value
        else:
            gap = 0
        # Format of Parameter:
        #  0: Param name
        #  1: Param class
        #  2: Param current value
        #  3: Param minimum value
        #  4: Param maximum value
        #  5: Param default value
        opt_tol = ilp.getParamInfo("MIPGap")
        opt_tol = opt_tol[2]
        # Exceed the tolerance slightly in case of floating point errors
        assert gap < 1.01 * opt_tol or timed_out or user_bd, "Unexplained bound mismatch"

        return cls(
            ex_id=ex_id,
            is_relax=relax,
            is_lower=is_lower,
            obj_val=obj_value,
            low_bound=low_bound,
            runtime=runtime,
            timed_out=timed_out,
            is_single=is_single,
        )

    @classmethod
    def create_noop(cls, val: int, ex_id: int, is_lower: bool, is_single: bool) -> "GurobiResult":
        r""" Create a no-op execution """
        return cls(
            ex_id=ex_id,
            is_lower=is_lower,
            is_relax=False,
            obj_val=val,
            low_bound=val,
            runtime=0,
            timed_out=False,
            is_single=is_single,
        )

    def log(self, header: str, is_debug: bool = False, bound_dist_str: str = "") -> NoReturn:
        r""" Log information about the gurobi results """
        log = logging.debug if is_debug else logging.info
        if not is_debug:
            header = f"{header} Final"
        perturb_format = "d" if not self.is_relax else ".6f"
        log(f"{header} - Ex-ID: {self.ex_id}")
        log(f"{header} - Is Lower Range Bound: {self.is_lower}")
        log(f"{header} - Is Relaxed?: {self.is_relax}")
        log(f"{header} - Is Single Cover?: {self.is_single}")
        log(f"{header} - Objective Low Bound: {self.low_bound:{perturb_format}}")
        log(f"{header} - Objective Value: {self.obj_val:{perturb_format}}")

        if self.obj_val > 0:
            gap = (self.obj_val - self.low_bound) / self.obj_val
        else:
            gap = 0
        log(f"{header} - Objective Gap: {gap:.3%}")

        log(f"{header} - Timed out?: {self.timed_out}")
        log(f"{header} - Runtime (s): {self.runtime:.6f}")

        if bound_dist_str:
            log(f"{header} - Gurobi Res. Bound Dist: {bound_dist_str}")


class GurobiStatusCode(enum.Enum):
    r"""
    Gurobi status codes. Reference:
    https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html
    """
    LOADED = 1
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    CUTOFF = 6
    ITERATION_LIMIT = 7
    NODE_LIMIT = 8
    TIME_LIMIT = 9
    SOLUTION_LIMIT = 10
    INTERRUPTED = 11
    NUMERIC = 12
    SUBOPTIMAL = 13
    INPROGRESS = 14
    USER_OBJ_LIMIT = 15
    WORK_LIMIT = 16

    @classmethod
    def get_valid(cls) -> "Set[GurobiStatusCode]":
        r""" Returns the set of valid status codes """
        return {cls.TIME_LIMIT, cls.OPTIMAL, cls.USER_OBJ_LIMIT}
