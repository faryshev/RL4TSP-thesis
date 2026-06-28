from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("loading POMO training modules...", flush=True)
    from rl4tsp.artifacts import save_json
    from rl4tsp.train import configure_torch_threads_from_env, build_model, save_checkpoint, set_seed, train_pomo

    configure_torch_threads_from_env()
    output_root = Path(config["output_root"])
    set_seed(config["seed"])
    print("building attention model for POMO", flush=True)
    model = build_model("attention", hidden_dim=config["hidden_dim"])
    print("starting POMO training", flush=True)
    trained, log = train_pomo(
        model,
        seed=config["seed"],
        n_train=config["n_train"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        lr=config["learning_rate"],
        num_starts=config["num_starts"],
        scale=config["scale"],
        device=config["device"],
        eval_interval=config.get("eval_interval", 1),
        progress=bool(config.get("progress", True)),
        progress_desc="POMO",
    )
    best_entry = min(log, key=lambda row: row.get("best_metric", float("inf"))) if log else {}
    save_checkpoint(
        output_root / "models" / "attention_pomo.pt",
        trained,
        {
            "model": "attention",
            "training": "pomo",
            "best_epoch": best_entry.get("best_epoch"),
            "best_metric_name": best_entry.get("best_metric_name"),
            "best_metric": best_entry.get("best_metric"),
            "config": config,
        },
    )
    save_json(output_root / "logs" / "attention_pomo_log.json", log)
    print("saved attention_pomo")


if __name__ == "__main__":
    main()
