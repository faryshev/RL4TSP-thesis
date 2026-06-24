from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("loading REINFORCE training modules...", flush=True)
    from rl4tsp.artifacts import save_json
    from rl4tsp.train import configure_torch_threads_from_env, build_model, save_checkpoint, set_seed, train_reinforce

    configure_torch_threads_from_env()
    output_root = Path(config["output_root"])
    log_dir = output_root / "logs"
    model_dir = output_root / "models"
    for model_name in config["models"]:
        for baseline_name in config["baselines"]:
            print(f"preparing {model_name}:{baseline_name}", flush=True)
            set_seed(config["seed"])
            model = build_model(model_name, hidden_dim=config["hidden_dim"])
            print(f"starting {model_name}:{baseline_name} training", flush=True)
            trained, log = train_reinforce(
                model,
                seed=config["seed"],
                n_train=config["n_train"],
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                lr=config["learning_rate"],
                baseline_name=baseline_name,
                scale=config["scale"],
                device=config["device"],
                progress=bool(config.get("progress", True)),
                progress_desc=f"{model_name}:{baseline_name}",
            )
            stem = f"{model_name}_{baseline_name}"
            best_entry = min(log, key=lambda row: row.get("best_metric", float("inf"))) if log else {}
            save_checkpoint(
                model_dir / f"{stem}.pt",
                trained,
                {
                    "model": model_name,
                    "training": "reinforce",
                    "baseline": baseline_name,
                    "best_epoch": best_entry.get("best_epoch"),
                    "best_metric_name": best_entry.get("best_metric_name"),
                    "best_metric": best_entry.get("best_metric"),
                    "config": config,
                },
            )
            save_json(log_dir / f"{stem}_log.json", log)
            print(f"saved {stem}")


if __name__ == "__main__":
    main()
