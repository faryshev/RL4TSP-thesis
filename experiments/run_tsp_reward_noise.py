from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("loading reward-noise modules...", flush=True)
    from rl4tsp.reward_noise import run_reward_noise_experiment
    from rl4tsp.train import configure_torch_threads_from_env

    configure_torch_threads_from_env()
    _, summary = run_reward_noise_experiment(config)
    print(summary)


if __name__ == "__main__":
    main()
