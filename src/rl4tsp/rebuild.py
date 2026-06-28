from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl4tsp.artifacts import add_sem_ci95_columns, build_summary, save_json
from rl4tsp.diagnostics import (
    _pearson_stats,
    _spearman_stats,
    plot_entropy_from_csv,
    plot_noise_summary_from_csv,
    plot_permutation_summary_from_csv,
)
from rl4tsp.plots import plot_tsp_summary
from rl4tsp.reward_noise import plot_reward_noise_outputs, summarize_reward_noise_raw


def _load_raw(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _require_columns(frame: pd.DataFrame, columns: list[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def summarize_permutation_raw(raw: pd.DataFrame) -> pd.DataFrame:
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
    return add_sem_ci95_columns(summary, ["length_abs_diff", "hamming"])


def summarize_noise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["sigma", "noise_percent", "method"], as_index=False)
        .agg(gap_mean=("gap_percent", "mean"), gap_std=("gap_percent", "std"), count=("gap_percent", "size"))
        .fillna(0.0)
    )
    return add_sem_ci95_columns(summary, ["gap"])


def summarize_entropy_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    if "sample_length_std" not in raw.columns and "sample_length_variance" in raw.columns:
        raw["sample_length_std"] = np.sqrt(raw["sample_length_variance"].clip(lower=0.0))
    _require_columns(
        raw,
        [
            "method",
            "n_cities",
            "mean_entropy",
            "logprob_gap",
            "sample_length_variance",
            "sample_length_std",
            "gap_percent",
        ],
        source="entropy_experiment_raw.csv",
    )
    rows: list[dict[str, Any]] = []
    for (method, n_cities), subset in raw.groupby(["method", "n_cities"]):
        entropy_pearson = _pearson_stats(subset["mean_entropy"], subset["gap_percent"])
        sharpness_pearson = _pearson_stats(subset["logprob_gap"], subset["gap_percent"])
        variance_pearson = _pearson_stats(subset["sample_length_variance"], subset["gap_percent"])
        entropy_spearman = _spearman_stats(subset["mean_entropy"], subset["gap_percent"])
        sharpness_spearman = _spearman_stats(subset["logprob_gap"], subset["gap_percent"])
        variance_spearman = _spearman_stats(subset["sample_length_variance"], subset["gap_percent"])
        rows.append(
            {
                "method": method,
                "n_cities": n_cities,
                "entropy_pearson_r": entropy_pearson["r"],
                "entropy_p_value": entropy_pearson["p"],
                "entropy_r_ci_low": entropy_pearson["ci_low"],
                "entropy_r_ci_high": entropy_pearson["ci_high"],
                "logprob_gap_pearson_r": sharpness_pearson["r"],
                "logprob_gap_p_value": sharpness_pearson["p"],
                "logprob_gap_r_ci_low": sharpness_pearson["ci_low"],
                "logprob_gap_r_ci_high": sharpness_pearson["ci_high"],
                "sample_variance_pearson_r": variance_pearson["r"],
                "sample_variance_p_value": variance_pearson["p"],
                "sample_variance_r_ci_low": variance_pearson["ci_low"],
                "sample_variance_r_ci_high": variance_pearson["ci_high"],
                "entropy_spearman_r": entropy_spearman["r"],
                "entropy_spearman_p_value": entropy_spearman["p"],
                "entropy_spearman_r_ci_low": entropy_spearman["ci_low"],
                "entropy_spearman_r_ci_high": entropy_spearman["ci_high"],
                "logprob_gap_spearman_r": sharpness_spearman["r"],
                "logprob_gap_spearman_p_value": sharpness_spearman["p"],
                "logprob_gap_spearman_r_ci_low": sharpness_spearman["ci_low"],
                "logprob_gap_spearman_r_ci_high": sharpness_spearman["ci_high"],
                "sample_variance_spearman_r": variance_spearman["r"],
                "sample_variance_spearman_p_value": variance_spearman["p"],
                "sample_variance_spearman_r_ci_low": variance_spearman["ci_low"],
                "sample_variance_spearman_r_ci_high": variance_spearman["ci_high"],
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
    summary = pd.DataFrame(rows).fillna(0.0)
    return add_sem_ci95_columns(summary, ["gap", "entropy", "logprob_gap", "sample_length_variance", "sample_length_std"])


def rebuild_scaling(run_dir: Path, figures_dir: Path) -> dict[str, str]:
    experiment_dir = run_dir / "tsp_comparison"
    raw_path = experiment_dir / "raw.csv"
    summary_path = experiment_dir / "summary.csv"
    written: dict[str, str] = {}
    raw = _load_raw(raw_path)
    if raw is not None:
        summary = build_summary(raw, group_cols=["n_cities", "method"], value_cols=["length", "gap_percent", "time_ms"])
        _write_csv(summary_path, summary)
        written["tsp_comparison_summary"] = str(summary_path)
    if summary_path.exists():
        for path in plot_tsp_summary(summary_path, figures_dir):
            written[path.name] = str(path)
    return written


def rebuild_diagnostics(run_dir: Path, figures_dir: Path) -> dict[str, str]:
    experiment_dir = run_dir / "tsp_diagnostics"
    written: dict[str, str] = {}

    permutation_raw = _load_raw(experiment_dir / "permutation_invariance_raw.csv")
    if permutation_raw is not None:
        summary_path = experiment_dir / "permutation_invariance_summary.csv"
        _write_csv(summary_path, summarize_permutation_raw(permutation_raw))
        plot_path = figures_dir / "permutation_invariance.png"
        plot_permutation_summary_from_csv(summary_path, plot_path)
        written["permutation_summary"] = str(summary_path)
        written["permutation_plot"] = str(plot_path)

    noise_raw = _load_raw(experiment_dir / "noise_robustness_raw.csv")
    if noise_raw is not None:
        summary_path = experiment_dir / "noise_robustness_summary.csv"
        summary = summarize_noise_raw(noise_raw)
        _write_csv(summary_path, summary)
        save_json(
            experiment_dir / "noise_robustness.json",
            {"raw": noise_raw.to_dict(orient="records"), "summary": summary.to_dict(orient="records")},
        )
        plot_path = figures_dir / "noise_robustness.png"
        plot_noise_summary_from_csv(summary_path, plot_path)
        written["noise_summary"] = str(summary_path)
        written["noise_json"] = str(experiment_dir / "noise_robustness.json")
        written["noise_plot"] = str(plot_path)

    entropy_raw = _load_raw(experiment_dir / "entropy_experiment_raw.csv")
    if entropy_raw is not None:
        summary_path = experiment_dir / "entropy_experiment_summary.csv"
        summary = summarize_entropy_raw(entropy_raw)
        _write_csv(summary_path, summary)
        save_json(
            experiment_dir / "entropy_experiment.json",
            {"raw": entropy_raw.to_dict(orient="records"), "summary": summary.to_dict(orient="records")},
        )
        plot_paths = plot_entropy_from_csv(experiment_dir / "entropy_experiment_raw.csv", summary_path, figures_dir)
        written["entropy_summary"] = str(summary_path)
        written["entropy_json"] = str(experiment_dir / "entropy_experiment.json")
        written.update(plot_paths)

    if written:
        save_json(
            experiment_dir / "metadata_rebuilt.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "written": written,
                "source": "recomputed from existing raw CSV files",
            },
        )
        written["diagnostics_rebuild_metadata"] = str(experiment_dir / "metadata_rebuilt.json")
    return written


def rebuild_reward_noise(run_dir: Path, figures_dir: Path) -> dict[str, str]:
    experiment_dir = run_dir / "tsp_reward_noise"
    raw_path = experiment_dir / "raw.csv"
    raw = _load_raw(raw_path)
    if raw is None:
        return {}
    summary_path = experiment_dir / "summary.csv"
    summary = summarize_reward_noise_raw(raw)
    _write_csv(summary_path, summary)
    plot_paths = plot_reward_noise_outputs(raw_path, summary_path, figures_dir)
    written = {
        "reward_noise_summary": str(summary_path),
        **{path.name: str(path) for path in plot_paths},
    }
    save_json(
        experiment_dir / "metadata_rebuilt.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "written": written,
            "source": "recomputed from existing raw CSV files",
        },
    )
    written["reward_noise_rebuild_metadata"] = str(experiment_dir / "metadata_rebuilt.json")
    return written


def rebuild_run_artifacts(run_dir: Path) -> dict[str, str]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    written.update(rebuild_scaling(run_dir, figures_dir))
    written.update(rebuild_diagnostics(run_dir, figures_dir))
    written.update(rebuild_reward_noise(run_dir, figures_dir))
    manifest_path = run_dir / "rebuild_manifest.json"
    save_json(
        manifest_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "written": written,
        },
    )
    written["rebuild_manifest"] = str(manifest_path)
    return written
