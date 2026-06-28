from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl4tsp.rebuild import rebuild_run_artifacts


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_rebuild_run_artifacts_recomputes_summaries_and_figures(tmp_path: Path):
    run_dir = tmp_path / "results" / "run" / "full"

    _write(
        run_dir / "tsp_comparison" / "raw.csv",
        [
            {"n_cities": 5, "instance": 0, "method": "A", "length": 10.0, "gap_percent": 1.0, "time_ms": 2.0},
            {"n_cities": 5, "instance": 1, "method": "A", "length": 12.0, "gap_percent": 3.0, "time_ms": 4.0},
        ],
    )
    _write(
        run_dir / "tsp_diagnostics" / "permutation_invariance_raw.csv",
        [
            {"method": "A", "length_abs_diff": 0.0, "hamming_distance": 0.0, "perfect_match": True},
            {"method": "A", "length_abs_diff": 1.0, "hamming_distance": 0.5, "perfect_match": False},
        ],
    )
    _write(
        run_dir / "tsp_diagnostics" / "noise_robustness_raw.csv",
        [
            {"sigma": 0.0, "noise_percent": 0.0, "method": "A", "gap_percent": 1.0},
            {"sigma": 0.1, "noise_percent": 1.0, "method": "A", "gap_percent": 2.0},
        ],
    )
    _write(
        run_dir / "tsp_diagnostics" / "entropy_experiment_raw.csv",
        [
            {
                "method": "A",
                "n_cities": 5,
                "instance": idx,
                "mean_entropy": float(idx),
                "logprob_gap": float(idx + 1),
                "sample_length_variance": float(idx * idx + 1),
                "sample_length_std": float(idx + 1),
                "gap_percent": float(idx + 2),
            }
            for idx in range(4)
        ],
    )
    _write(
        run_dir / "tsp_reward_noise" / "raw.csv",
        [
            {
                "phase": "eval",
                "sigma_rel": 0.0,
                "epoch": 1,
                "method": "A",
                "n_cities": 5,
                "length": 10.0,
                "gap_percent": 1.0,
                "time_ms": 2.0,
            },
            {
                "phase": "eval",
                "sigma_rel": 0.0,
                "epoch": 1,
                "method": "A",
                "n_cities": 5,
                "length": 12.0,
                "gap_percent": 3.0,
                "time_ms": 4.0,
            },
            {
                "phase": "train",
                "sigma_rel": 0.0,
                "epoch": 1,
                "method": "A",
                "n_cities": 5,
                "true_length_mean": 11.0,
                "observed_length_mean": 11.5,
            },
        ],
    )

    written = rebuild_run_artifacts(run_dir)

    assert (run_dir / "tsp_comparison" / "summary.csv").exists()
    assert (run_dir / "tsp_diagnostics" / "entropy_experiment_summary.csv").exists()
    assert (run_dir / "tsp_reward_noise" / "summary.csv").exists()
    entropy_summary = pd.read_csv(run_dir / "tsp_diagnostics" / "entropy_experiment_summary.csv")
    assert "entropy_spearman_r" in entropy_summary.columns
    for figure in [
        "tsp_gap.png",
        "tsp_time.png",
        "permutation_invariance.png",
        "noise_robustness.png",
        "entropy_gap.png",
        "entropy_correlations.png",
        "entropy_spearman_correlations.png",
        "reward_noise_gap.png",
        "reward_noise_training.png",
    ]:
        assert (run_dir / "figures" / figure).exists()
    assert Path(written["rebuild_manifest"]).exists()
