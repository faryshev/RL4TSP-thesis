from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

from rl4tsp.artifacts import build_summary, save_experiment_outputs
from rl4tsp.pomo import pomo_decode_best
from rl4tsp.progress import iter_progress
from rl4tsp.tsp import reference_tour, tour_length_numpy


def decode_greedy(model: nn.Module, points: np.ndarray) -> tuple[list[int], float]:
    import torch

    coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            actions, _, _ = model(coords, decode_mode="greedy")
    finally:
        model.train(was_training)
    route = actions.squeeze(0).cpu().numpy().astype(int).tolist()
    return route, tour_length_numpy(points, route)


def decode_pomo(model: nn.Module, points: np.ndarray, num_starts: int | None = None) -> tuple[list[int], float]:
    import torch

    coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    actions, _ = pomo_decode_best(model, coords, num_starts=num_starts)
    route = actions.squeeze(0).cpu().numpy().astype(int).tolist()
    return route, tour_length_numpy(points, route)


def run_tsp_comparison(
    *,
    models: dict[str, nn.Module],
    pomo_model: nn.Module | None,
    output_root: Path,
    experiment_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config["seed"])
    rows = []
    tasks = [
        (int(n_cities), int(instance))
        for n_cities in config["sizes"]
        for instance in range(config["instances_per_size"])
    ]
    for n_cities, instance in iter_progress(
        tasks,
        enabled=bool(config.get("progress", False)),
        total=len(tasks),
        desc="TSP comparison",
        unit="instance",
    ):
        points = rng.random((n_cities, 2)) * config["scale"]
        start = time.perf_counter()
        ref_route, ref_length, solver = reference_tour(
            points,
            exact_max_n=config["exact_max_n"],
            solver=config.get("reference_solver", "exact_or_nn_2opt"),
            lkh_scale=config.get("lkh_scale", 1_000_000),
        )
        reference_method = solver if solver == "LKH" else f"Reference:{solver}"
        rows.append(
            {
                "n_cities": n_cities,
                "instance": instance,
                "method": reference_method,
                "length": ref_length,
                "gap_percent": 0.0,
                "time_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
        for name, model in models.items():
            start = time.perf_counter()
            route, length = decode_greedy(model, points)
            rows.append(
                {
                    "n_cities": n_cities,
                    "instance": instance,
                    "method": name,
                    "length": length,
                    "gap_percent": (length - ref_length) / ref_length * 100.0,
                    "time_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
            assert sorted(route) == list(range(n_cities))
        if pomo_model is not None:
            start = time.perf_counter()
            pomo_route, pomo_length = decode_pomo(pomo_model, points, num_starts=config.get("pomo_num_starts"))
            rows.append(
                {
                    "n_cities": n_cities,
                    "instance": instance,
                    "method": "POMO",
                    "length": pomo_length,
                    "gap_percent": (pomo_length - ref_length) / ref_length * 100.0,
                    "time_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
            assert sorted(pomo_route) == list(range(n_cities))
        assert sorted(ref_route) == list(range(n_cities))
    raw = pd.DataFrame(rows)
    summary = build_summary(raw, group_cols=["n_cities", "method"], value_cols=["length", "gap_percent", "time_ms"])
    save_experiment_outputs(
        root=output_root,
        experiment_name=experiment_name,
        config=config,
        raw=raw,
        summary=summary,
        metadata={
            "methods": sorted(raw["method"].unique()),
            "model_checkpoints": config.get("model_checkpoints", []),
            "pomo_checkpoint": config.get("pomo_checkpoint"),
        },
    )
    return raw, summary
