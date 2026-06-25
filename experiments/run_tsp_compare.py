from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_required_checkpoint(path: str | None, model_name: str, hidden_dim: int):
    from rl4tsp.train import load_checkpoint

    if not path:
        raise ValueError(f"missing checkpoint path for {model_name}")
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"configured checkpoint does not exist: {checkpoint}")
    return load_checkpoint(checkpoint, model_name=model_name, hidden_dim=hidden_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("loading TSP comparison modules...", flush=True)
    from rl4tsp.benchmark import run_tsp_comparison
    from rl4tsp.train import configure_torch_threads_from_env

    configure_torch_threads_from_env()
    model_entries = config.get("model_checkpoints")
    if not model_entries:
        raise ValueError("config must define non-empty model_checkpoints")
    models = {}
    for entry in model_entries:
        method = entry["method"]
        if method in models:
            raise ValueError(f"duplicate comparison method: {method}")
        models[method] = _load_required_checkpoint(entry.get("checkpoint"), entry["model"], config["hidden_dim"])
    pomo_model = _load_required_checkpoint(
        config.get("pomo_checkpoint"),
        "attention",
        config["hidden_dim"],
    )
    output_root = Path(config["output_root"])
    experiment_name = config["experiment_name"]
    _, summary = run_tsp_comparison(
        models=models,
        pomo_model=pomo_model,
        output_root=output_root,
        experiment_name=experiment_name,
        config=config,
    )
    summary_csv = output_root / experiment_name / "summary.csv"
    if not summary_csv.exists():
        raise RuntimeError(f"TSP comparison finished but did not create expected summary CSV: {summary_csv}")
    print(f"wrote {summary_csv}")
    print(summary)


if __name__ == "__main__":
    main()
