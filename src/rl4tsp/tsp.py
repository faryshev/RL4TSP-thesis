from __future__ import annotations

import itertools
import importlib
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch


def generate_tsp_batch(batch_size: int, n_cities: int, scale: float = 10.0, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.rand(batch_size, n_cities, 2, device=device) * scale


def compute_tour_length(coords: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if tours.ndim == 1:
        tours = tours.unsqueeze(0)
    tours = tours.long().to(coords.device)
    batch_index = torch.arange(coords.size(0), device=coords.device).unsqueeze(1)
    route = coords[batch_index, tours]
    route_next = torch.cat([route[:, 1:], route[:, :1]], dim=1)
    return torch.norm(route - route_next, dim=-1).sum(dim=-1)


def tour_length_numpy(points: np.ndarray, tour: Iterable[int]) -> float:
    tour_array = np.asarray(list(tour), dtype=int)
    shifted = np.roll(tour_array, -1)
    return float(np.linalg.norm(points[tour_array] - points[shifted], axis=1).sum())


def validate_tour(tour: Iterable[int], n_cities: int) -> bool:
    tour_array = np.asarray(list(tour), dtype=int)
    return tour_array.shape == (n_cities,) and np.array_equal(np.sort(tour_array), np.arange(n_cities))


def normalize_tour(tour: Iterable[int], n_cities: int) -> list[int]:
    route = [int(node) for node in tour]
    if len(route) == n_cities + 1 and route[0] == route[-1]:
        route = route[:-1]
    if sorted(route) == list(range(1, n_cities + 1)):
        route = [node - 1 for node in route]
    if not validate_tour(route, n_cities):
        raise ValueError(f"solver returned an invalid tour for {n_cities} cities: {route}")
    return route


def distance_matrix(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def exact_tsp_dynamic_programming(points: np.ndarray) -> tuple[list[int], float]:
    n = len(points)
    if n == 0:
        raise ValueError("TSP instance must contain at least one city")
    if n == 1:
        return [0], 0.0
    dist = distance_matrix(points)

    @lru_cache(maxsize=None)
    def solve(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
        if mask == (1 << 0) | (1 << last):
            return float(dist[0, last]), (0, last)
        previous_mask = mask & ~(1 << last)
        best_cost = float("inf")
        best_path: tuple[int, ...] = ()
        for previous in range(1, n):
            if previous == last or not (previous_mask & (1 << previous)):
                continue
            cost, path = solve(previous_mask, previous)
            candidate = cost + float(dist[previous, last])
            if candidate < best_cost:
                best_cost = candidate
                best_path = (*path, last)
        return best_cost, best_path

    full_mask = (1 << n) - 1
    best_total = float("inf")
    best_route: tuple[int, ...] = ()
    for last in range(1, n):
        cost, path = solve(full_mask, last)
        total = cost + float(dist[last, 0])
        if total < best_total:
            best_total = total
            best_route = path
    return list(best_route), best_total


def nearest_neighbor_tour(points: np.ndarray, start: int = 0) -> list[int]:
    n = len(points)
    dist = distance_matrix(points)
    unvisited = set(range(n))
    current = start
    route = [current]
    unvisited.remove(current)
    while unvisited:
        current = min(unvisited, key=lambda node: dist[current, node])
        route.append(current)
        unvisited.remove(current)
    return route


def two_opt(points: np.ndarray, route: list[int], max_passes: int = 20) -> list[int]:
    best = route[:]
    best_length = tour_length_numpy(points, best)
    n = len(best)
    for _ in range(max_passes):
        improved = False
        for i, j in itertools.combinations(range(1, n), 2):
            if j - i == 1:
                continue
            candidate = best[:i] + best[i:j][::-1] + best[j:]
            candidate_length = tour_length_numpy(points, candidate)
            if candidate_length + 1e-12 < best_length:
                best, best_length = candidate, candidate_length
                improved = True
        if not improved:
            break
    return best


def nn_2opt_tour(points: np.ndarray) -> tuple[list[int], float]:
    starts = [0, max(0, len(points) // 3), max(0, 2 * len(points) // 3)]
    candidates = [two_opt(points, nearest_neighbor_tour(points, start=s)) for s in starts]
    route = min(candidates, key=lambda candidate: tour_length_numpy(points, candidate))
    return route, tour_length_numpy(points, route)


def lkh_tour(points: np.ndarray, scale_factor: int = 1_000_000) -> tuple[list[int], float]:
    try:
        elkai = importlib.import_module("elkai")
    except ImportError as exc:
        raise RuntimeError(
            "LKH reference requires the optional Python package `elkai`; "
            "install it or choose reference_solver='exact_or_nn_2opt'."
        ) from exc
    integer_distances = np.rint(distance_matrix(points) * scale_factor).astype(int).tolist()
    route = normalize_tour(elkai.solve_int_matrix(integer_distances), len(points))
    return route, tour_length_numpy(points, route)


def reference_tour(
    points: np.ndarray,
    exact_max_n: int = 10,
    solver: str = "exact_or_nn_2opt",
    lkh_scale: int = 1_000_000,
) -> tuple[list[int], float, str]:
    if solver not in {"exact_dp", "lkh", "nn_2opt", "exact_or_lkh", "exact_or_nn_2opt"}:
        raise ValueError(
            "solver must be one of: exact_dp, lkh, nn_2opt, exact_or_lkh, exact_or_nn_2opt"
        )
    if solver in {"exact_dp", "exact_or_lkh", "exact_or_nn_2opt"} and len(points) <= exact_max_n:
        route, length = exact_tsp_dynamic_programming(points)
        return route, length, "exact_dp"
    if solver == "exact_dp":
        raise ValueError(f"exact_dp requested for {len(points)} cities, above exact_max_n={exact_max_n}")
    if solver in {"lkh", "exact_or_lkh"}:
        route, length = lkh_tour(points, scale_factor=lkh_scale)
        return route, length, "LKH"
    route, length = nn_2opt_tour(points)
    return route, length, "nn_2opt"
