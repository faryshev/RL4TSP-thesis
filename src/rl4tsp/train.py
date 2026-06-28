from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl4tsp.artifacts import save_json
from rl4tsp.baselines import get_baseline
from rl4tsp.models import AttentionModel, PointerNet
from rl4tsp.pomo import pomo_decode_best, pomo_training_loss
from rl4tsp.progress import iter_progress
from rl4tsp.tsp import generate_tsp_batch


def configure_torch_threads_from_env() -> None:
    raw = os.environ.get("RL4TSP_TORCH_THREADS")
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        return
    torch.set_num_threads(threads)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name: str, hidden_dim: int = 128) -> nn.Module:
    if name == "attention":
        n_heads = next(heads for heads in range(min(8, hidden_dim), 0, -1) if hidden_dim % heads == 0)
        return AttentionModel(d_model=hidden_dim, n_heads=n_heads, num_layers=3)
    if name == "pointer":
        return PointerNet(hidden_dim=hidden_dim)
    raise ValueError(f"unknown model {name!r}")


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_reinforce(
    model: nn.Module,
    *,
    seed: int,
    n_train: int,
    batch_size: int,
    epochs: int,
    lr: float,
    baseline_name: str,
    scale: float = 10.0,
    device: str = "cpu",
    progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    set_seed(seed)
    model.to(device)
    model.train()
    baseline = get_baseline(baseline_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = 0
    epoch_iter = iter_progress(
        range(1, epochs + 1),
        enabled=progress,
        total=epochs,
        desc=progress_desc or f"REINFORCE:{baseline.name}",
        unit="epoch",
    )
    for epoch in epoch_iter:
        coords = generate_tsp_batch(batch_size, n_train, scale=scale, device=device)
        _, sampled_lengths, log_probs = model(coords, decode_mode="sample")
        if log_probs is None:
            raise RuntimeError("sample decoding did not return log probabilities")
        was_training = model.training
        model.eval()
        with torch.no_grad():
            try:
                _, greedy_lengths, _ = model(coords, decode_mode="greedy")
            finally:
                model.train(was_training)
        greedy_mean = float(greedy_lengths.mean().detach().cpu().item())
        is_best = greedy_mean < best_metric
        if is_best:
            best_metric = greedy_mean
            best_epoch = epoch
            best_state = _clone_state_dict(model)
        advantages = baseline.advantages(sampled_lengths, greedy_lengths).detach()
        loss = (advantages * log_probs.sum(dim=-1)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "sampled_mean_length": float(sampled_lengths.mean().detach().cpu().item()),
                "greedy_mean_length": greedy_mean,
                "baseline": baseline.name,
                "best_metric_name": "greedy_mean_length",
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "is_best": is_best,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), log


def train_pomo(
    model: nn.Module,
    *,
    seed: int,
    n_train: int,
    batch_size: int,
    epochs: int,
    lr: float,
    num_starts: int | None,
    scale: float = 10.0,
    device: str = "cpu",
    eval_interval: int | None = 1,
    progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if eval_interval is not None and eval_interval <= 0:
        raise ValueError("eval_interval must be positive or None")
    set_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = 0
    epoch_iter = iter_progress(
        range(1, epochs + 1),
        enabled=progress,
        total=epochs,
        desc=progress_desc or "POMO",
        unit="epoch",
    )
    for epoch in epoch_iter:
        coords = generate_tsp_batch(batch_size, n_train, scale=scale, device=device)
        loss, metrics = pomo_training_loss(model, coords, num_starts=num_starts)
        sample_best_mean = float(metrics["pomo_sample_best_length"])
        is_best = sample_best_mean < best_metric
        if is_best:
            best_metric = sample_best_mean
            best_epoch = epoch
            best_state = _clone_state_dict(model)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        should_eval = eval_interval is None or epoch == 1 or epoch == epochs or epoch % eval_interval == 0
        greedy_best_mean: float | None = None
        if should_eval:
            with torch.no_grad():
                _, best_lengths = pomo_decode_best(model, coords, num_starts=num_starts)
            greedy_best_mean = float(best_lengths.mean().detach().cpu().item())
        log.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "pomo_greedy_best_mean_length": greedy_best_mean,
                "best_metric_name": "pomo_sample_best_length",
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "is_best": is_best,
                **metrics,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), log


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    save_json(path.with_suffix(".json"), metadata)
    return path


def load_checkpoint(path: Path, model_name: str, hidden_dim: int = 128) -> nn.Module:
    model = build_model(model_name, hidden_dim=hidden_dim)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return model.eval()
