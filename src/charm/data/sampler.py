"""Batch sampler balanced across dentition types."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[list[int]]):
    """Draw a fixed number of samples of every group in each batch.

    Every batch contains ``samples_per_group[g]`` items of group ``g``. An
    epoch lasts until the largest group, relative to its share of the batch,
    has been seen once; smaller groups are reshuffled and cycled meanwhile,
    which upsamples rare dentition types.
    """

    def __init__(
        self, groups: Sequence[Sequence[int]], samples_per_group: Sequence[int]
    ):
        if len(groups) != len(samples_per_group):
            raise ValueError("One sample count is needed per group")
        active = [
            (list(group), int(k))
            for group, k in zip(groups, samples_per_group)
            if k > 0
        ]
        empty = [i for i, (group, _) in enumerate(active) if not group]
        if empty:
            raise ValueError("Groups with samples per batch must be nonempty")
        if not active:
            raise ValueError("At least one group must contribute samples")
        self.groups = active
        self.batch_size = sum(k for _, k in active)
        self.num_batches = max(math.ceil(len(g) / k) for g, k in active)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator().manual_seed(seed)
        queues = [[] for _ in self.groups]

        def draw(position: int, group: list[int]) -> int:
            if not queues[position]:
                order = torch.randperm(len(group), generator=generator)
                queues[position] = [group[i] for i in order.tolist()]
            return queues[position].pop()

        for _ in range(self.num_batches):
            batch = []
            for position, (group, k) in enumerate(self.groups):
                batch.extend(draw(position, group) for _ in range(k))
            yield batch
