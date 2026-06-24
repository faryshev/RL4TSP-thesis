from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchGreedyMeanBaseline:
    """Subtract the batch mean greedy length from sampled tour lengths."""

    name: str = "batch_greedy_mean"

    def advantages(self, sampled_lengths: torch.Tensor, greedy_lengths: torch.Tensor) -> torch.Tensor:
        return sampled_lengths - greedy_lengths.mean()


@dataclass(frozen=True)
class GreedyRolloutBaseline:
    """Per-instance greedy rollout baseline matching the common NCO formulation."""

    name: str = "greedy_rollout"

    def advantages(self, sampled_lengths: torch.Tensor, greedy_lengths: torch.Tensor) -> torch.Tensor:
        return sampled_lengths - greedy_lengths


BASELINES = {
    BatchGreedyMeanBaseline.name: BatchGreedyMeanBaseline(),
    GreedyRolloutBaseline.name: GreedyRolloutBaseline(),
}


def get_baseline(name: str) -> BatchGreedyMeanBaseline | GreedyRolloutBaseline:
    try:
        return BASELINES[name]
    except KeyError as exc:
        known = ", ".join(sorted(BASELINES))
        raise ValueError(f"unknown baseline {name!r}; expected one of: {known}") from exc
