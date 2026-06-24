from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rl4tsp.artifacts import build_summary, save_experiment_outputs
from rl4tsp.pomo import POMOConfig, expand_pomo_batch, pomo_decode_best, pomo_group_advantage, resolve_num_starts
from rl4tsp.plot_style import PALETTE, add_ci95_note, apply_figure_style, style_axis, style_for
from rl4tsp.progress import iter_progress
from rl4tsp.train import build_model, save_checkpoint, set_seed
from rl4tsp.tsp import generate_tsp_batch, reference_tour, tour_length_numpy


POSITIVE_FLOOR = 1e-8


@dataclass(frozen=True)
class NoisyAdvantage:
    sampled_observed: torch.Tensor
    baseline_observed: torch.Tensor
    advantage: torch.Tensor


def apply_relative_reward_noise(
    true_lengths: torch.Tensor,
    *,
    sigma_rel: float,
    generator: torch.Generator | None = None,
    positive_floor: float = POSITIVE_FLOOR,
) -> torch.Tensor:
    if sigma_rel < 0:
        raise ValueError("sigma_rel must be nonnegative")
    if positive_floor <= 0:
        raise ValueError("positive_floor must be positive")
    if sigma_rel == 0:
        return true_lengths.clone()
    eps = torch.randn(
        true_lengths.shape,
        dtype=true_lengths.dtype,
        device=true_lengths.device,
        generator=generator,
    ) * sigma_rel
    return torch.clamp(true_lengths * (1.0 + eps), min=positive_floor)


def compute_noisy_advantage(
    sampled_true: torch.Tensor,
    baseline_true: torch.Tensor,
    *,
    sigma_rel: float,
    baseline_name: str,
    generator: torch.Generator | None = None,
    positive_floor: float = POSITIVE_FLOOR,
) -> NoisyAdvantage:
    sampled_observed = apply_relative_reward_noise(
        sampled_true,
        sigma_rel=sigma_rel,
        generator=generator,
        positive_floor=positive_floor,
    )
    baseline_observed = apply_relative_reward_noise(
        baseline_true,
        sigma_rel=sigma_rel,
        generator=generator,
        positive_floor=positive_floor,
    )
    if baseline_name == "greedy_rollout":
        advantage = sampled_observed - baseline_observed
    elif baseline_name == "batch_greedy_mean":
        advantage = sampled_observed - baseline_observed.mean()
    else:
        raise ValueError("baseline_name must be 'greedy_rollout' or 'batch_greedy_mean'")
    return NoisyAdvantage(sampled_observed, baseline_observed, advantage)


def _make_generator(seed: int, device: str | torch.device) -> torch.Generator:
    try:
        generator = torch.Generator(device=torch.device(device))
    except RuntimeError:
        generator = torch.Generator()
    return generator.manual_seed(seed)


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def train_reinforce_with_reward_noise(
    model: nn.Module,
    *,
    seed: int,
    n_train: int,
    train_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    baseline_name: str,
    sigma_rel: float,
    scale: float = 10.0,
    device: str = "cpu",
    positive_floor: float = POSITIVE_FLOOR,
    progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if train_size <= 1:
        raise ValueError("train_size must be greater than 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(seed)
    noise_generator = _make_generator(seed + 1009, device)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches_per_epoch = math.ceil(n_train / batch_size)
    log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = 0

    epoch_iter = iter_progress(
        range(1, epochs + 1),
        enabled=progress,
        total=epochs,
        desc=progress_desc or f"reward-noise sigma={sigma_rel:g}",
        unit="epoch",
    )
    for epoch in epoch_iter:
        losses: list[float] = []
        true_means: list[float] = []
        observed_means: list[float] = []
        baseline_true_means: list[float] = []
        baseline_observed_means: list[float] = []
        weights: list[int] = []
        epoch_is_best = False

        for batch_index in range(batches_per_epoch):
            current_batch = min(batch_size, n_train - batch_index * batch_size)
            coords = generate_tsp_batch(current_batch, train_size, scale=scale, device=device)
            _, sampled_true, log_probs = model(coords, decode_mode="sample")
            if log_probs is None:
                raise RuntimeError("sample decoding did not return log probabilities")

            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    _, baseline_true, _ = model(coords, decode_mode="greedy")
            finally:
                model.train(was_training)

            batch_true_mean = float(sampled_true.mean().detach().cpu().item())
            if batch_true_mean < best_metric:
                best_metric = batch_true_mean
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                epoch_is_best = True
            noisy = compute_noisy_advantage(
                sampled_true,
                baseline_true,
                sigma_rel=sigma_rel,
                baseline_name=baseline_name,
                generator=noise_generator,
                positive_floor=positive_floor,
            )
            loss = (noisy.advantage.detach() * log_probs.sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            weight = int(current_batch)
            weights.append(weight)
            losses.append(float(loss.detach().cpu().item()))
            true_means.append(batch_true_mean)
            observed_means.append(float(noisy.sampled_observed.mean().detach().cpu().item()))
            baseline_true_means.append(float(baseline_true.mean().detach().cpu().item()))
            baseline_observed_means.append(float(noisy.baseline_observed.mean().detach().cpu().item()))

        true_length_mean = _weighted_mean(true_means, weights)
        log.append(
            {
                "phase": "train",
                "epoch": epoch,
                "sigma_rel": float(sigma_rel),
                "loss": _weighted_mean(losses, weights),
                "true_length_mean": true_length_mean,
                "observed_length_mean": _weighted_mean(observed_means, weights),
                "baseline_true_mean": _weighted_mean(baseline_true_means, weights),
                "baseline_observed_mean": _weighted_mean(baseline_observed_means, weights),
                "baseline": baseline_name,
                "n_cities": train_size,
                "best_metric_name": "true_length_mean",
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "is_best": epoch_is_best,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), log


def train_pomo_with_reward_noise(
    model: nn.Module,
    *,
    seed: int,
    n_train: int,
    train_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    num_starts: int | None,
    sigma_rel: float,
    scale: float = 10.0,
    device: str = "cpu",
    positive_floor: float = POSITIVE_FLOOR,
    progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if train_size <= 1:
        raise ValueError("train_size must be greater than 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(seed)
    noise_generator = _make_generator(seed + 2003, device)
    starts_count = resolve_num_starts(num_starts, train_size)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches_per_epoch = math.ceil(n_train / batch_size)
    log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = 0

    epoch_iter = iter_progress(
        range(1, epochs + 1),
        enabled=progress,
        total=epochs,
        desc=progress_desc or f"POMO reward-noise sigma={sigma_rel:g}",
        unit="epoch",
    )
    for epoch in epoch_iter:
        losses: list[float] = []
        true_means: list[float] = []
        observed_means: list[float] = []
        weights: list[int] = []
        epoch_is_best = False

        for batch_index in range(batches_per_epoch):
            current_batch = min(batch_size, n_train - batch_index * batch_size)
            coords = generate_tsp_batch(current_batch, train_size, scale=scale, device=device)
            expanded, starts = expand_pomo_batch(coords, POMOConfig(starts_count))
            _, true_lengths, log_probs = model(expanded, decode_mode="sample", fixed_start=starts)
            if log_probs is None:
                raise RuntimeError("POMO reward-noise training requires sampled log probabilities")
            observed_lengths = apply_relative_reward_noise(
                true_lengths,
                sigma_rel=sigma_rel,
                generator=noise_generator,
                positive_floor=positive_floor,
            )
            grouped_true = true_lengths.detach().view(current_batch, starts_count)
            grouped_observed = observed_lengths.detach().view(current_batch, starts_count)
            batch_true_best = float(grouped_true.min(dim=1).values.mean().cpu().item())
            if batch_true_best < best_metric:
                best_metric = batch_true_best
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                epoch_is_best = True
            advantage = pomo_group_advantage(observed_lengths.detach(), current_batch, starts_count)
            loss = (advantage * log_probs.sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            weight = int(current_batch)
            weights.append(weight)
            losses.append(float(loss.detach().cpu().item()))
            true_means.append(batch_true_best)
            observed_means.append(float(grouped_observed.min(dim=1).values.mean().cpu().item()))

        true_length_mean = _weighted_mean(true_means, weights)
        log.append(
            {
                "phase": "train",
                "epoch": epoch,
                "sigma_rel": float(sigma_rel),
                "loss": _weighted_mean(losses, weights),
                "true_length_mean": true_length_mean,
                "observed_length_mean": _weighted_mean(observed_means, weights),
                "baseline_true_mean": math.nan,
                "baseline_observed_mean": math.nan,
                "baseline": "pomo_group_mean",
                "n_cities": train_size,
                "best_metric_name": "pomo_true_best_mean_length",
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "is_best": epoch_is_best,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), log


def evaluate_model_true(
    model: nn.Module,
    *,
    config: dict[str, Any],
    sigma_rel: float,
    epoch: int,
    method: str,
    decoder: str = "greedy",
    eval_cases: list[dict[str, Any]] | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if eval_cases is None:
        eval_cases = build_reward_noise_eval_cases(config, progress=progress)

    was_training = model.training
    model.eval()
    try:
        for case in iter_progress(
            eval_cases,
            enabled=progress,
            total=len(eval_cases),
            desc=f"eval {method} sigma={float(sigma_rel):g}",
            unit="instance",
        ):
            points = case["points"]
            n_cities = int(case["n_cities"])
            instance = int(case["instance"])
            ref_route = case["reference_route"]
            ref_length = float(case["reference_length"])
            solver = str(case["reference_solver"])
            start = time.perf_counter()
            coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if decoder == "pomo":
                    actions, _ = pomo_decode_best(
                        model,
                        coords,
                        num_starts=config.get("pomo_num_starts", config.get("num_starts")),
                    )
                else:
                    actions, _, _ = model(coords, decode_mode="greedy")
            route = actions.squeeze(0).cpu().numpy().astype(int).tolist()
            length = tour_length_numpy(points, route)
            rows.append(
                {
                    "phase": "eval",
                    "sigma_rel": float(sigma_rel),
                    "epoch": int(epoch),
                    "method": method,
                    "n_cities": n_cities,
                    "instance": instance,
                    "length": length,
                    "reference_length": ref_length,
                    "gap_percent": (length - ref_length) / ref_length * 100.0,
                    "reference_solver": solver,
                    "time_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
            assert sorted(route) == list(range(n_cities))
            assert sorted(ref_route) == list(range(n_cities))
    finally:
        model.train(was_training)
    return rows


def build_reward_noise_eval_cases(config: dict[str, Any], *, progress: bool = False) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config["seed"])
    eval_sizes = config.get("eval_sizes", config.get("sizes"))
    if not eval_sizes:
        raise ValueError("config must define eval_sizes")
    eval_instances = int(config.get("eval_instances", config.get("instances_per_size", 1)))
    tasks = [(int(n_cities), int(instance)) for n_cities in eval_sizes for instance in range(eval_instances)]
    cases: list[dict[str, Any]] = []
    for n_cities, instance in iter_progress(
        tasks,
        enabled=progress,
        total=len(tasks),
        desc="reward-noise reference set",
        unit="instance",
    ):
        points = rng.random((n_cities, 2)) * config.get("scale", 10.0)
        ref_route, ref_length, solver = reference_tour(
            points,
            exact_max_n=int(config.get("exact_max_n", 10)),
            solver=config.get("reference_solver", "exact_or_nn_2opt"),
            lkh_scale=int(config.get("lkh_scale", 1_000_000)),
        )
        cases.append(
            {
                "n_cities": n_cities,
                "instance": instance,
                "points": points,
                "reference_route": ref_route,
                "reference_length": ref_length,
                "reference_solver": solver,
            }
        )
    return cases


def plot_reward_noise_outputs(raw_csv: Path, summary_csv: Path, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    eval_summary = summary[summary["phase"] == "eval"].copy()
    if not eval_summary.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        apply_figure_style(fig)
        for (method, n_cities), subset in eval_summary.groupby(["method", "n_cities"]):
            subset = subset.sort_values("sigma_rel")
            yerr_col = "gap_percent_ci95" if "gap_percent_ci95" in subset else "gap_percent_std"
            style = style_for(method)
            yerr = subset[yerr_col]
            ax.errorbar(
                subset["sigma_rel"],
                subset["gap_percent_mean"],
                yerr=yerr,
                capsize=4,
                elinewidth=1.2,
                capthick=1.2,
                label=f"{method}, n={int(n_cities)}",
                **style,
            )
        ax.set_xlabel("Относительное стандартное отклонение шума награды")
        ax.set_ylabel("Отклонение при истинной оценке, %")
        ax.set_title("Качество при шумной обучающей награде: среднее и 95% ДИ")
        ax.legend(fontsize=8)
        style_axis(ax)
        add_ci95_note(ax)
        fig.tight_layout()
        path = output_dir / "reward_noise_gap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    train_raw = raw[raw["phase"] == "train"].copy()
    if not train_raw.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        apply_figure_style(fig)
        short_method = {
            "Attention:greedy_rollout": "Attn",
            "PointerNet:greedy_rollout": "Ptr",
            "POMO": "POMO",
        }
        for (method, sigma_rel), subset in train_raw.groupby(["method", "sigma_rel"]):
            subset = subset.sort_values("epoch")
            label_prefix = f"{short_method.get(method, method)}, σ={sigma_rel:g}"
            style = style_for(method)
            ax.plot(
                subset["epoch"],
                subset["true_length_mean"],
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                color=style.get("color", PALETTE["reinforce"]),
                label=f"{label_prefix}, ист.",
            )
            ax.plot(
                subset["epoch"],
                subset["observed_length_mean"],
                marker="x",
                linestyle="--",
                color=style.get("color", PALETTE["reinforce"]),
                label=f"{label_prefix}, шум.",
            )
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Длина тура в обучении")
        ax.set_title("Динамика истинной и зашумленной обучающей цели")
        ax.set_yscale("log")
        ax.legend(
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            columnspacing=0.8,
            handlelength=1.8,
            labelspacing=0.25,
            borderaxespad=0.2,
        )
        style_axis(ax)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.38)
        path = output_dir / "reward_noise_training.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        written.append(path)

    return written


def run_reward_noise_experiment(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = Path(config["output_root"])
    experiment_name = config.get("experiment_name", "tsp_reward_noise")
    figures_dir = output_root / "figures"
    model_names = config.get("models", [config.get("model", "attention")])
    baselines = config.get("baselines", [config.get("baseline", "greedy_rollout")])
    sigma_values = config.get("sigma_rel_values", [config.get("sigma_rel", 0.0)])
    rows: list[dict[str, Any]] = []
    checkpoint_payloads: list[dict[str, Any]] = []
    experiment_dir = output_root / experiment_name
    model_labels = {"attention": "Attention", "pointer": "PointerNet", "pomo": "POMO"}
    progress = bool(config.get("progress", False))
    print("building reward-noise evaluation reference set", flush=True)
    eval_cases = build_reward_noise_eval_cases(config, progress=progress)

    for model_name in model_names:
        baseline_loop = [None] if model_name == "pomo" else baselines
        for baseline_name in baseline_loop:
            method = "POMO" if model_name == "pomo" else f"{model_labels.get(model_name, model_name)}:{baseline_name}"
            for sigma_rel in sigma_values:
                print(f"preparing reward-noise run: {method}, sigma={float(sigma_rel):g}", flush=True)
                set_seed(int(config["seed"]))
                if model_name == "pomo":
                    model = build_model("attention", hidden_dim=int(config["hidden_dim"]))
                    trained, train_log = train_pomo_with_reward_noise(
                        model,
                        seed=int(config["seed"]),
                        n_train=int(config["n_train"]),
                        train_size=int(config["train_size"]),
                        batch_size=int(config["batch_size"]),
                        epochs=int(config["epochs"]),
                        lr=float(config["learning_rate"]),
                        num_starts=config.get("pomo_num_starts", config.get("num_starts")),
                        sigma_rel=float(sigma_rel),
                        scale=float(config.get("scale", 10.0)),
                        device=config.get("device", "cpu"),
                        progress=bool(config.get("progress", False)),
                        progress_desc=f"reward-noise {method} sigma={float(sigma_rel):g}",
                    )
                    decoder = "pomo"
                    checkpoint_model = "attention"
                    stem = f"pomo_sigma_{float(sigma_rel):g}".replace(".", "p")
                else:
                    model = build_model(model_name, hidden_dim=int(config["hidden_dim"]))
                    trained, train_log = train_reinforce_with_reward_noise(
                        model,
                        seed=int(config["seed"]),
                        n_train=int(config["n_train"]),
                        train_size=int(config["train_size"]),
                        batch_size=int(config["batch_size"]),
                        epochs=int(config["epochs"]),
                        lr=float(config["learning_rate"]),
                        baseline_name=str(baseline_name),
                        sigma_rel=float(sigma_rel),
                        scale=float(config.get("scale", 10.0)),
                        device=config.get("device", "cpu"),
                        progress=bool(config.get("progress", False)),
                        progress_desc=f"reward-noise {method} sigma={float(sigma_rel):g}",
                    )
                    decoder = "greedy"
                    checkpoint_model = model_name
                    stem = f"{model_name}_{baseline_name}_sigma_{float(sigma_rel):g}".replace(".", "p")
                checkpoint_path = experiment_dir / "models" / f"{stem}.pt"
                best_entry = min(train_log, key=lambda row: row.get("best_metric", float("inf"))) if train_log else {}
                save_checkpoint(
                    checkpoint_path,
                    trained,
                    {
                        "model": checkpoint_model,
                        "method": method,
                        "training": "reward_noise",
                        "sigma_rel": float(sigma_rel),
                        "best_epoch": best_entry.get("best_epoch"),
                        "best_metric_name": best_entry.get("best_metric_name"),
                        "best_metric": best_entry.get("best_metric"),
                        "config": config,
                    },
                )
                checkpoint_payloads.append({"method": method, "sigma_rel": float(sigma_rel), "checkpoint": str(checkpoint_path)})
                for entry in train_log:
                    rows.append({"method": method, "checkpoint": str(checkpoint_path), **entry})
                rows.extend(
                    evaluate_model_true(
                        trained,
                        config=config,
                        sigma_rel=float(sigma_rel),
                        epoch=int(config["epochs"]),
                        method=method,
                        decoder=decoder,
                        eval_cases=eval_cases,
                        progress=progress,
                    )
                )

    raw = pd.DataFrame(rows)
    for column in [
        "observed_length_mean",
        "true_length_mean",
        "baseline_observed_mean",
        "baseline_true_mean",
        "length",
        "reference_length",
        "gap_percent",
        "time_ms",
    ]:
        if column not in raw:
            raw[column] = np.nan
    summary = build_summary(
        raw,
        group_cols=["phase", "sigma_rel", "epoch", "method", "n_cities"],
        value_cols=[
            "observed_length_mean",
            "true_length_mean",
            "baseline_observed_mean",
            "length",
            "gap_percent",
            "time_ms",
        ],
    )
    outputs = save_experiment_outputs(
        root=output_root,
        experiment_name=experiment_name,
        config=config,
        raw=raw,
        summary=summary,
        metadata={
            "experiment": "tsp_training_reward_noise",
            "methods": sorted(raw["method"].dropna().unique().tolist()),
            "sigma_rel_values": [float(value) for value in sigma_values],
            "evaluation_objective": "true_tour_length",
            "training_objective": "observed_noisy_tour_length",
            "evaluation_reference_set": "precomputed once per run and reused for every method/sigma variant",
            "checkpoints": checkpoint_payloads,
            "figures": {
                "reward_noise_gap.png": str(figures_dir / "reward_noise_gap.png"),
                "reward_noise_training.png": str(figures_dir / "reward_noise_training.png"),
            },
            "plot_sources": {
                "reward_noise_gap.png": {"summary_csv": str((output_root / experiment_name / "summary.csv"))},
                "reward_noise_training.png": {
                    "raw_csv": str((output_root / experiment_name / "raw.csv")),
                    "summary_csv": str((output_root / experiment_name / "summary.csv")),
                },
            },
            "intervals": "95% confidence intervals for aggregated plots when repeated observations are available",
        },
    )
    plot_reward_noise_outputs(outputs.raw_csv, outputs.summary_csv, figures_dir)
    return raw, summary
