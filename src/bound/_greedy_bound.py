__all__ = [
    "calc",
]

import collections
import dataclasses
import heapq
import math
from typing import List, NoReturn, Set

from torch import LongTensor

GreedyBound = collections.namedtuple("GreedyBound", "low_bound greedy_val")


@dataclasses.dataclass
class DatasetBlock:
    id_num: int
    cur_pos: int = -1
    mods: Set[int] = dataclasses.field(default_factory=set)

    def del_models(self, mods: Set[int]) -> NoReturn:
        r""" Remove the models from the set """
        orig_len = len(self)
        self.mods.difference_update(mods)
        # Should only ever delete if at least one model uses the block
        assert orig_len > len(self)

    def get_models(self) -> Set[int]:
        return self.mods

    def add_model(self, mod_id: int) -> None:
        self.mods.add(mod_id)

    def __len__(self) -> int:
        return len(self.mods)

    def __lt__(self, other):
        r""" Uses a min queue so prioritize the bigger example"""
        return len(self) > len(other)


def _calc_low_bound_approx_ratio(perturb_dist: int, spread_max: int) -> float:
    r"""
    Calculates the lower bound approximation ratio.

    See Theorem 4 in Slavik's paper "Tight Analysis of Greedy Set Covering"

    :param perturb_dist: Perturbation distance
    :param spread_max: Maximum spread degree
    :return: Approximation ratio for the lower bound
    """
    terms = [
        math.log(perturb_dist),
        -math.log(math.log(perturb_dist)),
        3,
        math.log(math.log(32)),
        -math.log(32),
    ]
    harmonic = sum(1 / i for i in range(1, spread_max + 1))
    return min(sum(terms), harmonic)


def _sift_down(heap: List[DatasetBlock], block: DatasetBlock) -> NoReturn:
    r"""
    The \p heapq class does not implement a useful siftdown operation that supports updating
    the position of nodes so implementing a custom sift down
    """
    assert block.cur_pos >= 0, "Block position seems unset"
    assert block.cur_pos < len(heap), "Current position overflowed the end"
    assert block == heap[block.cur_pos], "Block position in heap became out of sync"

    cur_pos = block.cur_pos
    while True:
        new_pos, best = None, block
        left_child = 2 * cur_pos + 1
        right_child = left_child + 1
        for idx in [left_child, right_child]:
            if idx < len(heap) and heap[idx] < best:
                assert heap[idx].cur_pos == idx, "Block position in heap became out of sync"
                new_pos, best = idx, heap[idx]

        if new_pos is None:
            break
        else:
            # Update the item in the blocks old location
            heap[cur_pos] = heap[new_pos]
            heap[cur_pos].cur_pos = cur_pos
            # No need to write into new position here. Do it once at the end.
            # Otherwise new_pos will get written in the next iteration
            # Update the position iteration
            cur_pos = new_pos
    # Update the block's current position
    heap[cur_pos] = block
    heap[cur_pos].cur_pos = cur_pos


def _heap_pop(heap) -> DatasetBlock:
    r"""
    Pop the top element from the heap
    """
    block = heap[0]
    if len(heap) > 1:
        heap[0] = heap.pop()
        heap[0].cur_pos = 0
        _sift_down(heap, heap[0])
    else:
        heap.pop()
    return block


def _greedy_select(model_to_blocks: List[Set[int]], all_blocks: List[DatasetBlock],
                   perturb_dist: int) -> Set[int]:
    r""" Performs the actual greedy selection """
    # Create the heap and record the positions
    heap = [_block for _block in all_blocks if len(_block) > 0]
    heapq.heapify(heap)
    # Record the starting postion of each block in the heap. May be changed as blocks are updated
    # moved around the heap
    for cur_pos, _block in enumerate(heap):
        pos_bl = all_blocks[_block.id_num]
        assert pos_bl.cur_pos == -1, "Block's position already set"
        pos_bl.cur_pos = cur_pos

    # Continue perturbing elements until the bound is crossed
    perturbed_blocks, perturb_models = set(), set()
    while len(perturb_models) < perturb_dist:
        block = _heap_pop(heap)
        assert block.cur_pos == 0, "Unexpected top element in the heap"

        # Verify not perturbing the same block twice
        assert block.id_num not in perturbed_blocks, "Cannot perturb a block twice"
        perturbed_blocks.add(block.id_num)

        # Verify not perturbing the same model twice
        tmp_mods = block.get_models()
        assert len(perturb_models.intersection(tmp_mods)) == 0, "Trying to perturb model twice"
        perturb_models.update(tmp_mods)

        # Get the blocks that need to be updated
        blocks_to_update = {block_id for mod_id in tmp_mods
                            for block_id in model_to_blocks[mod_id]}
        blocks_to_update.difference_update(perturbed_blocks)
        for block_id in blocks_to_update:
            tmp_block = all_blocks[block_id]
            tmp_block.del_models(tmp_mods)

            _sift_down(heap=heap, block=tmp_block)

    assert len(perturb_models) >= perturb_dist, "Insufficient models perturbed"
    return perturbed_blocks


def calc(model, models_list: LongTensor, perturb_dist: int) -> GreedyBound:
    r"""

    :param model: Model being analyzed
    :param models_list:
    :param perturb_dist:
    :return: Greedy bound being analyzed
    """
    assert model.n_models & 1 == 1, "Calculations expect an odd number of models"

    # Handle the naive case where greedy and exact solution are identical
    if perturb_dist <= 1:
        return GreedyBound(low_bound=1, greedy_val=1)

    # Construct the base data structures containing the model info
    model_to_blocks = [None for _ in range(model.n_models)]
    all_blocks = [DatasetBlock(id_num=block_id) for block_id in range(model.n_ds_parts)]
    # Create the two way mapping between blocks and submodels
    sorted_models_list = sorted(models_list.tolist())
    for model_id in sorted_models_list:
        ds_blocks_lst = model.get_submodel_ds_parts(model_id)
        model_to_blocks[model_id] = set(ds_blocks_lst)
        for block_id in ds_blocks_lst:
            all_blocks[block_id].add_model(mod_id=model_id)

    # Calculate the maximum spread degree
    spread_max = max(len(block) for block in all_blocks)

    # Perform the greedy selection
    # noinspection PyTypeChecker
    perturbed_blocks = _greedy_select(model_to_blocks=model_to_blocks, all_blocks=all_blocks,
                                      perturb_dist=perturb_dist)
    greedy_bound = len(perturbed_blocks)

    approx_factor = _calc_low_bound_approx_ratio(perturb_dist=perturb_dist, spread_max=spread_max)
    bound = int(math.ceil(greedy_bound / approx_factor))
    assert bound >= 1, "Bound cannot be zero"

    return GreedyBound(low_bound=bound, greedy_val=greedy_bound)
