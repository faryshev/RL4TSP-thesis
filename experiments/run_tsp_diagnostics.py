from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_entries(entries: list[dict], hidden_dim: int) -> list[dict]:
    from rl4tsp.train import load_checkpoint

    loaded = []
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"configured checkpoint does not exist: {checkpoint}")
        loaded.append(
            {
                **entry,
                "model_object": load_checkpoint(checkpoint, model_name=entry["model"], hidden_dim=hidden_dim),
            }
        )
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("loading diagnostic modules...", flush=True)
    from rl4tsp.diagnostics import (
        run_entropy_experiment,
        run_noise_robustness,
        run_permutation_experiment,
        save_diagnostics_outputs,
    )
    from rl4tsp.train import configure_torch_threads_from_env

    configure_torch_threads_from_env()
    output_root = Path(config["output_root"])
    output_dir = output_root / config.get("experiment_name", "tsp_diagnostics")
    figures_dir = output_root / "figures"
    hidden_dim = config["hidden_dim"]
    model_entries = _load_entries(config["model_checkpoints"], hidden_dim=hidden_dim)
    entropy_entries = _load_entries(config.get("entropy_model_checkpoints", []), hidden_dim=hidden_dim)

    permutation_raw = permutation_summary = None
    if config.get("run_permutation", True):
        permutation_raw, permutation_summary = run_permutation_experiment(
            model_entries,
            n_cities=config["permutation_n_cities"],
            num_instances=config["permutation_instances"],
            num_permutations=config["permutations_per_instance"],
            scale=config["scale"],
            seed=config["seed"],
            progress=bool(config.get("progress", True)),
        )

    noise_raw = noise_summary = None
    if config.get("run_noise", True):
        noise_raw, noise_summary = run_noise_robustness(
            model_entries,
            n_cities=config["noise_n_cities"],
            num_instances=config["noise_instances"],
            sigmas=config["noise_sigmas"],
            scale=config["scale"],
            seed=config["seed"],
            exact_max_n=config["exact_max_n"],
            reference_solver=config.get("reference_solver", "exact_or_nn_2opt"),
            lkh_scale=config.get("lkh_scale", 1_000_000),
            progress=bool(config.get("progress", True)),
        )

    entropy_raw = entropy_summary = None
    if config.get("run_entropy", True):
        if not entropy_entries:
            raise ValueError("run_entropy=True requires entropy_model_checkpoints")
        entropy_raw, entropy_summary = run_entropy_experiment(
            entropy_entries,
            n_cities_list=config["entropy_sizes"],
            num_instances=config["entropy_instances"],
            scale=config["scale"],
            seed=config["seed"],
            exact_max_n=config["exact_max_n"],
            reference_solver=config.get("reference_solver", "exact_or_nn_2opt"),
            lkh_scale=config.get("lkh_scale", 1_000_000),
            stochastic_samples=int(config.get("entropy_stochastic_samples", 8)),
            progress=bool(config.get("progress", True)),
        )

    written = save_diagnostics_outputs(
        output_dir=output_dir,
        figures_dir=figures_dir,
        config=config,
        permutation_raw=permutation_raw,
        permutation_summary=permutation_summary,
        noise_raw=noise_raw,
        noise_summary=noise_summary,
        entropy_raw=entropy_raw,
        entropy_summary=entropy_summary,
    )
    print(written)


if __name__ == "__main__":
    main()
