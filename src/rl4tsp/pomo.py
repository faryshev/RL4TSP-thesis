from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class POMOConfig:
    num_starts: int | None = None


def resolve_num_starts(num_starts: int | None, n_cities: int) -> int:
    starts_count = n_cities if num_starts is None else num_starts
    if starts_count <= 0:
        raise ValueError("num_starts must be positive")
    if starts_count > n_cities:
        raise ValueError("num_starts cannot exceed n_cities")
    return starts_count


def expand_pomo_batch(coords: torch.Tensor, config: POMOConfig) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_cities, _ = coords.shape
    num_starts = resolve_num_starts(config.num_starts, n_cities)
    starts = torch.arange(num_starts, device=coords.device).repeat(batch_size)
    expanded = coords.repeat_interleave(num_starts, dim=0)
    return expanded, starts


def pomo_group_advantage(lengths: torch.Tensor, batch_size: int, num_starts: int) -> torch.Tensor:
    grouped = lengths.view(batch_size, num_starts)
    return (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)


def pomo_decode_best(model: nn.Module, coords: torch.Tensor, num_starts: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    batch_size, n_cities, _ = coords.shape
    starts_count = resolve_num_starts(num_starts, n_cities)
    expanded, starts = expand_pomo_batch(coords, POMOConfig(starts_count))
    try:
        with torch.no_grad():
            actions, lengths, _ = model(expanded, decode_mode="greedy", fixed_start=starts)
    finally:
        model.train(was_training)
    grouped_lengths = lengths.view(batch_size, starts_count)
    best_indices = torch.argmin(grouped_lengths, dim=1)
    grouped_actions = actions.view(batch_size, starts_count, n_cities)
    best_actions = grouped_actions[torch.arange(batch_size, device=coords.device), best_indices]
    best_lengths = grouped_lengths[torch.arange(batch_size, device=coords.device), best_indices]
    return best_actions, best_lengths


def pomo_training_loss(model: nn.Module, coords: torch.Tensor, num_starts: int | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, n_cities, _ = coords.shape
    starts_count = resolve_num_starts(num_starts, n_cities)
    expanded, starts = expand_pomo_batch(coords, POMOConfig(starts_count))
    _, lengths, log_probs = model(expanded, decode_mode="sample", fixed_start=starts)
    if log_probs is None:
        raise RuntimeError("POMO training requires sampled decoding with log probabilities")
    advantage = pomo_group_advantage(lengths.detach(), batch_size, starts_count)
    loss = (advantage * log_probs.sum(dim=-1)).mean()
    metrics = {
        "pomo_sample_mean_length": float(lengths.mean().detach().cpu().item()),
        "pomo_sample_best_length": float(lengths.view(batch_size, starts_count).min(dim=1).values.mean().detach().cpu().item()),
    }
    return loss, metrics
