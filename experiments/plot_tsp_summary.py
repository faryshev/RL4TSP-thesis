from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".runtime-cache" / "matplotlib"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = Path(args.summary)
    if not summary.exists():
        raise FileNotFoundError(
            f"TSP summary CSV was not found: {summary}. "
            "Run experiments/run_tsp_compare.py first, or rerun run_experiments.sh from the project root."
        )
    print("loading plotting modules...", flush=True)
    from rl4tsp.plots import plot_tsp_summary

    written = plot_tsp_summary(summary, Path(args.output_dir))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
