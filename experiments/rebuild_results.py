from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".runtime-cache" / "matplotlib"))


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir)
    if args.run_root is not None:
        return Path(args.run_root) / args.scale
    raise ValueError("provide either positional RUN_DIR or --run-root")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild summary CSV files, correlation tables, JSON summaries and figures "
            "from existing raw experiment outputs. This script does not train models "
            "and does not recompute reference solvers."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Run scale directory, for example results/20260625_120000/full",
    )
    parser.add_argument(
        "--run-root",
        help="Run root directory, for example results/20260625_120000. Used with --scale.",
    )
    parser.add_argument("--scale", choices=["small", "full"], default="full")
    args = parser.parse_args()

    print("loading rebuild modules...", flush=True)
    from rl4tsp.rebuild import rebuild_run_artifacts

    run_dir = resolve_run_dir(args)
    written = rebuild_run_artifacts(run_dir)
    if not written:
        print(f"No rebuildable raw CSV files found under {run_dir}")
        return
    print(f"Rebuilt artifacts from: {run_dir}")
    for label, path in sorted(written.items()):
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
