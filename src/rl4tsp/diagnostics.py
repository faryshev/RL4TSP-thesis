from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rl4tsp.artifacts import add_sem_ci95_columns, save_json
from rl4tsp.benchmark import decode_greedy, decode_pomo
from rl4tsp.plot_style import METHOD_STYLES, PALETTE, add_ci95_note, apply_figure_style, style_axis
from rl4tsp.pomo import POMOConfig, expand_pomo_batch, resolve_num_starts
from rl4tsp.progress import iter_progress
from rl4tsp.tsp import reference_tour, tour_length_numpy, validate_tour

Decoder = Callable[[nn.Module, np.ndarray], tuple[list[int], float]]

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


def canonicalize_tour(tour: list[int] | np.ndarray) -> np.ndarray:
    route = np.asarray(tour, dtype=int)
    if route.ndim != 1 or len(route) == 0:
        raise ValueError("tour must be a non-empty one-dimensional sequence")
    start = int(np.argmin(route))
    shifted = np.roll(route, -start)
    if len(shifted) > 2 and shifted[1] > shifted[-1]:
        shifted = np.concatenate(([shifted[0]], shifted[:0:-1]))
    return shifted


def entropy_from_probs(probs: np.ndarray) -> float:
    positive = probs[probs > 0]
    return float(-(positive * np.log(positive)).sum())


def logprob_gap_from_probs(probs: np.ndarray) -> float:
    positive = np.sort(np.asarray(probs, dtype=float)[np.asarray(probs) > 0.0])[::-1]
    if len(positive) < 2:
        return 0.0
    return float(np.log(positive[0]) - np.log(positive[1]))


def _signals_from_probabilities(probabilities: list[torch.Tensor], row: int = 0) -> tuple[float, float]:
    entropy_values = []
    gap_values = []
    for step in probabilities:
        probs = step[row].numpy() if step.ndim == 2 else step.numpy()
        entropy_values.append(entropy_from_probs(probs))
        gap_values.append(logprob_gap_from_probs(probs))
    return float(np.mean(entropy_values)), float(np.mean(gap_values))


def _decoder_for(name: str, pomo_num_starts: int | None = None) -> Decoder:
    if name == "greedy":
        return decode_greedy
    if name == "pomo":
        return lambda model, points: decode_pomo(model, points, num_starts=pomo_num_starts)
    raise ValueError(f"unknown decoder {name!r}; expected 'greedy' or 'pomo'")


def run_permutation_experiment(
    model_entries: list[dict],
    *,
    n_cities: int,
    num_instances: int,
    num_permutations: int,
    scale: float,
    seed: int,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for entry in model_entries:
        model = entry["model_object"]
        method = entry["method"]
        decoder = _decoder_for(entry.get("decoder", "greedy"), entry.get("pomo_num_starts"))
        for instance_id in iter_progress(
            range(num_instances),
            enabled=progress,
            total=num_instances,
            desc=f"permutation:{method}",
            unit="instance",
        ):
            points = rng.random((n_cities, 2)) * scale
            base_tour, base_length = decoder(model, points)
            if not validate_tour(base_tour, n_cities):
                raise RuntimeError(f"{method} produced invalid base tour")
            base_canonical = canonicalize_tour(base_tour)
            for permutation_id in range(num_permutations):
                permutation = rng.permutation(n_cities)
                permuted_points = points[permutation]
                perm_tour, perm_length = decoder(model, permuted_points)
                if not validate_tour(perm_tour, n_cities):
                    raise RuntimeError(f"{method} produced invalid permuted tour")
                mapped_tour = permutation[np.asarray(perm_tour, dtype=int)]
                mapped_canonical = canonicalize_tour(mapped_tour)
                hamming_distance = float(np.mean(base_canonical != mapped_canonical))
                rows.append(
                    {
                        "method": method,
                        "instance": instance_id,
                        "permutation": permutation_id,
                        "length_abs_diff": abs(base_length - perm_length),
                        "hamming_distance": hamming_distance,
                        "perfect_match": hamming_distance < 1e-12,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("method", as_index=False)
        .agg(
            length_abs_diff_mean=("length_abs_diff", "mean"),
            length_abs_diff_std=("length_abs_diff", "std"),
            hamming_mean=("hamming_distance", "mean"),
            hamming_std=("hamming_distance", "std"),
            perfect_match_rate=("perfect_match", "mean"),
            count=("perfect_match", "size"),
        )
        .fillna(0.0)
    )
    summary = add_sem_ci95_columns(summary, ["length_abs_diff", "hamming"])
    return raw, summary


def run_noise_robustness(
    model_entries: list[dict],
    *,
    n_cities: int,
    num_instances: int,
    sigmas: list[float],
    scale: float,
    seed: int,
    exact_max_n: int,
    reference_solver: str,
    lkh_scale: int,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    instances = [rng.random((n_cities, 2)) * scale for _ in range(num_instances)]
    references = [
        reference_tour(points, exact_max_n=exact_max_n, solver=reference_solver, lkh_scale=lkh_scale)
        for points in instances
    ]
    rows = []
    for sigma in iter_progress(
        sigmas,
        enabled=progress,
        total=len(sigmas),
        desc="noise robustness",
        unit="sigma",
    ):
        for instance_id, (points, (_, ref_length, ref_solver)) in enumerate(zip(instances, references)):
            noisy_points = np.clip(points + rng.normal(0.0, sigma, size=points.shape), 0.0, scale)
            for entry in model_entries:
                method = entry["method"]
                decoder = _decoder_for(entry.get("decoder", "greedy"), entry.get("pomo_num_starts"))
                route, _ = decoder(entry["model_object"], noisy_points)
                if not validate_tour(route, n_cities):
                    raise RuntimeError(f"{method} produced invalid noisy tour")
                true_length = tour_length_numpy(points, route)
                rows.append(
                    {
                        "sigma": sigma,
                        "noise_percent": 100.0 * sigma / scale,
                        "instance": instance_id,
                        "method": method,
                        "length": true_length,
                        "reference_length": ref_length,
                        "reference_solver": ref_solver,
                        "gap_percent": (true_length - ref_length) / ref_length * 100.0,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["sigma", "noise_percent", "method"], as_index=False)
        .agg(gap_mean=("gap_percent", "mean"), gap_std=("gap_percent", "std"), count=("gap_percent", "size"))
        .fillna(0.0)
    )
    summary = add_sem_ci95_columns(summary, ["gap"])
    return raw, summary


def policy_rollout_signals(
    model: nn.Module,
    points: np.ndarray,
    *,
    decoder: str = "greedy",
    pomo_num_starts: int | None = None,
) -> tuple[list[int], float, float, float]:
    coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if decoder == "pomo":
                starts_count = resolve_num_starts(pomo_num_starts, points.shape[0])
                expanded, starts = expand_pomo_batch(coords, POMOConfig(starts_count))
                actions, lengths, _, probabilities = model(
                    expanded,
                    decode_mode="greedy",
                    fixed_start=starts,
                    return_probabilities=True,
                )
                best_idx = int(torch.argmin(lengths).detach().cpu().item())
                route = actions[best_idx].cpu().numpy().astype(int).tolist()
                entropy, logprob_gap = _signals_from_probabilities(probabilities[1:], row=best_idx)
                return route, tour_length_numpy(points, route), entropy, logprob_gap
            actions, lengths, _, probabilities = model(coords, decode_mode="greedy", return_probabilities=True)
    finally:
        model.train(was_training)
    route = actions.squeeze(0).cpu().numpy().astype(int).tolist()
    entropy, logprob_gap = _signals_from_probabilities(probabilities, row=0)
    return route, float(lengths.item()), entropy, logprob_gap


def attention_rollout_with_entropy(model: nn.Module, points: np.ndarray) -> tuple[list[int], float, float]:
    route, length, entropy, _ = policy_rollout_signals(model, points, decoder="greedy")
    return route, length, entropy


def stochastic_length_variance(
    model: nn.Module,
    points: np.ndarray,
    *,
    decoder: str = "greedy",
    pomo_num_starts: int | None = None,
    samples: int = 8,
) -> tuple[float, float]:
    if samples <= 1:
        return 0.0, 0.0
    coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    lengths: list[float] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(samples):
                if decoder == "pomo":
                    starts_count = resolve_num_starts(pomo_num_starts, points.shape[0])
                    expanded, starts = expand_pomo_batch(coords, POMOConfig(starts_count))
                    actions, sampled_lengths, _ = model(expanded, decode_mode="sample", fixed_start=starts)
                    best_idx = int(torch.argmin(sampled_lengths).detach().cpu().item())
                    route = actions[best_idx].cpu().numpy().astype(int).tolist()
                    lengths.append(tour_length_numpy(points, route))
                else:
                    actions, _, _ = model(coords, decode_mode="sample")
                    route = actions.squeeze(0).cpu().numpy().astype(int).tolist()
                    lengths.append(tour_length_numpy(points, route))
    finally:
        model.train(was_training)
    return float(np.var(lengths, ddof=1)), float(np.std(lengths, ddof=1))


def _pearson_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or clean["x"].std() == 0 or clean["y"].std() == 0:
        return {"r": 0.0, "p": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    if scipy_stats is None:
        r = float(clean["x"].corr(clean["y"]))
        p_value = 1.0
    else:
        result = scipy_stats.pearsonr(clean["x"], clean["y"])
        r = float(result.statistic)
        p_value = float(result.pvalue)
    if len(clean) <= 3 or abs(r) >= 1.0:
        return {"r": r, "p": p_value, "ci_low": r, "ci_high": r}
    z_value = np.arctanh(r)
    delta = 1.96 / math.sqrt(len(clean) - 3)
    return {
        "r": r,
        "p": p_value,
        "ci_low": float(np.tanh(z_value - delta)),
        "ci_high": float(np.tanh(z_value + delta)),
    }


def run_entropy_experiment(
    model_entries: list[dict],
    *,
    n_cities_list: list[int],
    num_instances: int,
    scale: float,
    seed: int,
    exact_max_n: int,
    reference_solver: str,
    lkh_scale: int,
    stochastic_samples: int = 8,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for entry in model_entries:
        method = entry["method"]
        model = entry["model_object"]
        decoder = entry.get("decoder", "greedy")
        pomo_num_starts = entry.get("pomo_num_starts")
        for n_cities in iter_progress(
            n_cities_list,
            enabled=progress,
            total=len(n_cities_list),
            desc=f"entropy:{method}",
            unit="size",
        ):
            for instance_id in range(num_instances):
                points = rng.random((n_cities, 2)) * scale
                route, length, mean_entropy, mean_logprob_gap = policy_rollout_signals(
                    model,
                    points,
                    decoder=decoder,
                    pomo_num_starts=pomo_num_starts,
                )
                if not validate_tour(route, n_cities):
                    raise RuntimeError(f"{method} produced invalid entropy tour")
                sample_variance, sample_std = stochastic_length_variance(
                    model,
                    points,
                    decoder=decoder,
                    pomo_num_starts=pomo_num_starts,
                    samples=stochastic_samples,
                )
                _, ref_length, ref_solver = reference_tour(
                    points,
                    exact_max_n=exact_max_n,
                    solver=reference_solver,
                    lkh_scale=lkh_scale,
                )
                rows.append(
                    {
                        "method": method,
                        "n_cities": n_cities,
                        "instance": instance_id,
                        "mean_entropy": mean_entropy,
                        "logprob_gap": mean_logprob_gap,
                        "sample_length_variance": sample_variance,
                        "sample_length_std": sample_std,
                        "model_length": length,
                        "reference_length": ref_length,
                        "reference_solver": ref_solver,
                        "gap_percent": (length - ref_length) / ref_length * 100.0,
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for (method, n_cities), subset in raw.groupby(["method", "n_cities"]):
        entropy_stats = _pearson_stats(subset["mean_entropy"], subset["gap_percent"])
        sharpness_stats = _pearson_stats(subset["logprob_gap"], subset["gap_percent"])
        variance_stats = _pearson_stats(subset["sample_length_variance"], subset["gap_percent"])
        summary_rows.append(
            {
                "method": method,
                "n_cities": n_cities,
                "entropy_pearson_r": entropy_stats["r"],
                "entropy_p_value": entropy_stats["p"],
                "entropy_r_ci_low": entropy_stats["ci_low"],
                "entropy_r_ci_high": entropy_stats["ci_high"],
                "logprob_gap_pearson_r": sharpness_stats["r"],
                "logprob_gap_p_value": sharpness_stats["p"],
                "logprob_gap_r_ci_low": sharpness_stats["ci_low"],
                "logprob_gap_r_ci_high": sharpness_stats["ci_high"],
                "sample_variance_pearson_r": variance_stats["r"],
                "sample_variance_p_value": variance_stats["p"],
                "sample_variance_r_ci_low": variance_stats["ci_low"],
                "sample_variance_r_ci_high": variance_stats["ci_high"],
                "count": len(subset),
                "gap_mean": float(subset["gap_percent"].mean()),
                "gap_std": float(subset["gap_percent"].std()),
                "entropy_mean": float(subset["mean_entropy"].mean()),
                "entropy_std": float(subset["mean_entropy"].std()),
                "logprob_gap_mean": float(subset["logprob_gap"].mean()),
                "logprob_gap_std": float(subset["logprob_gap"].std()),
                "sample_length_variance_mean": float(subset["sample_length_variance"].mean()),
                "sample_length_variance_std": float(subset["sample_length_variance"].std()),
                "sample_length_std_mean": float(subset["sample_length_std"].mean()),
                "sample_length_std_std": float(subset["sample_length_std"].std()),
            }
        )
    summary = pd.DataFrame(summary_rows).fillna(0.0)
    summary = add_sem_ci95_columns(summary, ["gap", "entropy", "logprob_gap", "sample_length_variance", "sample_length_std"])
    return raw, summary


def _style_for(method: str) -> dict:
    return METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-", "color": None})


def _draw_ci_band(ax, x, y, yerr, *, color: str | None) -> None:
    if yerr is None:
        return
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.14, linewidth=0)


def plot_permutation_summary(summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    apply_figure_style(fig)
    ordered = summary.sort_values("hamming_mean")
    ax.bar(ordered["method"], ordered["hamming_mean"], color=PALETTE["pointer"], yerr=ordered["hamming_ci95"], capsize=3)
    ax.set_ylabel("Средняя доля несовпавших позиций")
    ax.set_xlabel("Метод")
    ax.set_title("Эквивариантность к перестановке городов: среднее и 95% ДИ")
    ax.tick_params(axis="x", rotation=30)
    style_axis(ax, grid_axis="y")
    add_ci95_note(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_noise_summary(summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_figure_style(fig)
    for method in sorted(summary["method"].unique()):
        subset = summary[summary["method"] == method].sort_values("noise_percent")
        style = _style_for(method)
        yerr = subset["gap_ci95"] if "gap_ci95" in subset else subset["gap_std"]
        _draw_ci_band(ax, subset["noise_percent"], subset["gap_mean"], yerr, color=style.get("color"))
        ax.errorbar(
            subset["noise_percent"],
            subset["gap_mean"],
            yerr=yerr,
            capsize=4,
            elinewidth=1.2,
            capthick=1.2,
            label=method,
            **style,
        )
    ax.set_xlabel("Шум координат, % от масштаба")
    ax.set_ylabel("Среднее отклонение от опорного решения, %")
    ax.set_title("Устойчивость к шуму координат: среднее и 95% ДИ")
    ax.legend()
    style_axis(ax)
    add_ci95_note(ax, "Точки показывают среднее; вертикальные интервалы и заливка показывают 95% доверительный интервал.")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_entropy_raw(raw: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals = [
        ("mean_entropy", "Средняя энтропия"),
        ("logprob_gap", "Разрыв log p1 - log p2"),
        ("sample_length_variance", "Дисперсия длины при сэмплировании"),
    ]
    fig, axes = plt.subplots(1, len(signals), figsize=(6 * len(signals), 4.8), squeeze=False)
    apply_figure_style(fig)
    for ax, (column, label) in zip(axes.ravel(), signals):
        for method in sorted(raw["method"].unique()):
            method_subset = raw[raw["method"] == method]
            style = _style_for(method)
            ax.scatter(
                method_subset[column],
                method_subset["gap_percent"],
                s=16,
                alpha=0.5,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                label=method,
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Отклонение от опорного решения, %")
        ax.legend()
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_entropy_correlations(summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals = [
        ("entropy", "Энтропия"),
        ("logprob_gap", "Разрыв log p1 - log p2"),
        ("sample_variance", "Дисперсия сэмплов"),
    ]
    fig, axes = plt.subplots(1, len(signals), figsize=(6 * len(signals), 4.8), squeeze=False)
    apply_figure_style(fig)
    for ax, (prefix, title) in zip(axes.ravel(), signals):
        r_col = f"{prefix}_pearson_r"
        low_col = f"{prefix}_r_ci_low"
        high_col = f"{prefix}_r_ci_high"
        for method in sorted(summary["method"].unique()):
            subset = summary[summary["method"] == method].sort_values("n_cities")
            style = _style_for(method)
            lower = (subset[r_col] - subset[low_col]).clip(lower=0.0)
            upper = (subset[high_col] - subset[r_col]).clip(lower=0.0)
            ax.errorbar(
                subset["n_cities"],
                subset[r_col],
                yerr=np.vstack([lower.to_numpy(), upper.to_numpy()]),
                capsize=4,
                elinewidth=1.2,
                capthick=1.2,
                label=method,
                **style,
            )
        ax.axhline(0.5, color=PALETTE["grid"], linestyle="--", linewidth=1)
        ax.axhline(-0.5, color=PALETTE["grid"], linestyle="--", linewidth=1)
        ax.set_title(f"{title}: Pearson r и 95% ДИ")
        ax.set_xlabel("Число городов")
        ax.set_ylabel("Pearson r с отклонением")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        style_axis(ax)
        add_ci95_note(ax, "Вертикальные интервалы показывают 95% доверительный интервал для Pearson r.")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_permutation_summary_from_csv(summary_csv: Path, output_path: Path) -> Path:
    return plot_permutation_summary(pd.read_csv(summary_csv), output_path)


def plot_noise_summary_from_csv(summary_csv: Path, output_path: Path) -> Path:
    return plot_noise_summary(pd.read_csv(summary_csv), output_path)


def plot_entropy_from_csv(raw_csv: Path, summary_csv: Path, output_dir: Path) -> dict[str, str]:
    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)
    gap_path = output_dir / "entropy_gap.png"
    correlations_path = output_dir / "entropy_correlations.png"
    plot_entropy_raw(raw, summary, gap_path)
    plot_entropy_correlations(summary, correlations_path)
    return {
        "entropy_plot": str(gap_path),
        "entropy_correlations_plot": str(correlations_path),
    }


def save_diagnostics_outputs(
    *,
    output_dir: Path,
    figures_dir: Path | None = None,
    config: dict,
    permutation_raw: pd.DataFrame | None,
    permutation_summary: pd.DataFrame | None,
    noise_raw: pd.DataFrame | None,
    noise_summary: pd.DataFrame | None,
    entropy_raw: pd.DataFrame | None,
    entropy_summary: pd.DataFrame | None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is None:
        figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    save_json(output_dir / "config.json", config)
    written["config"] = str(output_dir / "config.json")
    plot_sources: dict[str, dict[str, str]] = {}
    figures: dict[str, str] = {}
    if permutation_raw is not None and permutation_summary is not None:
        raw_path = output_dir / "permutation_invariance_raw.csv"
        summary_path = output_dir / "permutation_invariance_summary.csv"
        plot_path = figures_dir / "permutation_invariance.png"
        permutation_raw.to_csv(raw_path, index=False)
        permutation_summary.to_csv(summary_path, index=False)
        plot_permutation_summary_from_csv(summary_path, plot_path)
        written["permutation_raw"] = str(raw_path)
        written["permutation_summary"] = str(summary_path)
        written["permutation_plot"] = str(plot_path)
        plot_sources["permutation_plot"] = {"summary_csv": str(summary_path)}
        figures[plot_path.name] = str(plot_path)
    if noise_raw is not None and noise_summary is not None:
        raw_path = output_dir / "noise_robustness_raw.csv"
        summary_path = output_dir / "noise_robustness_summary.csv"
        plot_path = figures_dir / "noise_robustness.png"
        noise_raw.to_csv(raw_path, index=False)
        noise_summary.to_csv(summary_path, index=False)
        save_json(
            output_dir / "noise_robustness.json",
            {"raw": noise_raw.to_dict(orient="records"), "summary": noise_summary.to_dict(orient="records")},
        )
        plot_noise_summary_from_csv(summary_path, plot_path)
        written["noise_raw"] = str(raw_path)
        written["noise_summary"] = str(summary_path)
        written["noise_json"] = str(output_dir / "noise_robustness.json")
        written["noise_plot"] = str(plot_path)
        plot_sources["noise_plot"] = {"summary_csv": str(summary_path)}
        figures[plot_path.name] = str(plot_path)
    if entropy_raw is not None and entropy_summary is not None:
        raw_path = output_dir / "entropy_experiment_raw.csv"
        summary_path = output_dir / "entropy_experiment_summary.csv"
        entropy_raw.to_csv(raw_path, index=False)
        entropy_summary.to_csv(summary_path, index=False)
        save_json(
            output_dir / "entropy_experiment.json",
            {"raw": entropy_raw.to_dict(orient="records"), "summary": entropy_summary.to_dict(orient="records")},
        )
        plot_paths = plot_entropy_from_csv(raw_path, summary_path, figures_dir)
        written["entropy_raw"] = str(raw_path)
        written["entropy_summary"] = str(summary_path)
        written["entropy_json"] = str(output_dir / "entropy_experiment.json")
        written.update(plot_paths)
        plot_sources["entropy_plot"] = {"raw_csv": str(raw_path), "summary_csv": str(summary_path)}
        plot_sources["entropy_correlations_plot"] = {"summary_csv": str(summary_path)}
        for path in plot_paths.values():
            path_obj = Path(path)
            figures[path_obj.name] = str(path_obj)
    save_json(
        output_dir / "metadata.json",
        {
            "written": written,
            "plot_sources": plot_sources,
            "figures": figures,
            "intervals": "95% confidence intervals for aggregated plots when repeated observations are available",
        },
    )
    written["metadata"] = str(output_dir / "metadata.json")
    return written
