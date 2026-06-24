import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import json

from rl4tsp import plots
from rl4tsp.artifacts import build_summary, save_experiment_outputs
from rl4tsp.baselines import BatchGreedyMeanBaseline, GreedyRolloutBaseline
from rl4tsp.diagnostics import (
    attention_rollout_with_entropy,
    canonicalize_tour,
    entropy_from_probs,
    logprob_gap_from_probs,
    plot_noise_summary,
    plot_permutation_summary,
    policy_rollout_signals,
    run_entropy_experiment,
    run_noise_robustness,
    run_permutation_experiment,
    save_diagnostics_outputs,
    stochastic_length_variance,
)
from rl4tsp.models import AttentionModel, PointerNet
from rl4tsp.pomo import POMOConfig, expand_pomo_batch, pomo_group_advantage
from rl4tsp.tsp import (
    compute_tour_length,
    exact_tsp_dynamic_programming,
    lkh_tour,
    normalize_tour,
    reference_tour,
    tour_length_numpy,
    validate_tour,
)


def test_tour_length_closes_cycle_and_exact_solver_handles_square():
    square = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tour = np.array([0, 1, 2, 3])

    torch_length = compute_tour_length(
        torch.tensor(square).unsqueeze(0),
        torch.tensor(tour).unsqueeze(0),
    ).item()
    numpy_length = tour_length_numpy(square, tour)
    exact_route, exact_length = exact_tsp_dynamic_programming(square)

    assert torch_length == 4.0
    assert numpy_length == 4.0
    assert exact_length == 4.0
    assert validate_tour(exact_route, 4)


def test_reference_tour_uses_exact_solver_for_small_instances():
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    route, length, solver_name = reference_tour(points, exact_max_n=8)

    assert solver_name == "exact_dp"
    assert validate_tour(route, 4)
    assert length == 8.0


def test_normalize_tour_accepts_repeated_return_and_one_indexed_routes():
    assert normalize_tour([0, 2, 1, 0], 3) == [0, 2, 1]
    assert normalize_tour([1, 3, 2, 1], 3) == [0, 2, 1]

    with pytest.raises(ValueError, match="invalid tour"):
        normalize_tour([0, 1, 1], 3)


def test_lkh_tour_uses_elkai_integer_distance_matrix(monkeypatch):
    class FakeElkai:
        matrix = None

        @staticmethod
        def solve_int_matrix(matrix):
            FakeElkai.matrix = matrix
            return [1, 2, 3, 1]

    monkeypatch.setitem(sys.modules, "elkai", FakeElkai)
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    route, length = lkh_tour(points, scale_factor=100)

    assert route == [0, 1, 2]
    assert FakeElkai.matrix[0][1] == 100
    assert length == pytest.approx(1.0 + 1.0 + np.sqrt(2.0))


def test_lkh_tour_missing_dependency_is_explicit(monkeypatch):
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "elkai":
            raise ImportError("missing elkai")
        return real_import_module(name, package=package)

    monkeypatch.setattr("rl4tsp.tsp.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="LKH reference requires"):
        lkh_tour(np.random.default_rng(0).random((4, 2)))


def test_reference_tour_uses_lkh_above_exact_limit_when_requested(monkeypatch):
    class FakeElkai:
        @staticmethod
        def solve_int_matrix(matrix):
            return [0, 1, 2, 3]

    monkeypatch.setitem(sys.modules, "elkai", FakeElkai)
    points = np.random.default_rng(0).random((4, 2)).astype(np.float32)

    route, length, solver_name = reference_tour(points, exact_max_n=3, solver="exact_or_lkh")

    assert solver_name == "LKH"
    assert validate_tour(route, 4)
    assert length == pytest.approx(tour_length_numpy(points, route))


def test_diagnostics_canonicalize_tour_is_rotation_and_direction_invariant():
    assert canonicalize_tour([2, 3, 0, 1]).tolist() == [0, 1, 2, 3]
    assert canonicalize_tour([2, 1, 0, 3]).tolist() == [0, 1, 2, 3]


def test_diagnostics_entropy_from_probabilities():
    probs = np.array([0.5, 0.5, 0.0])

    assert entropy_from_probs(probs) == pytest.approx(np.log(2.0))


def test_logprob_gap_from_probabilities_uses_top_two_actions():
    probs = np.array([0.7, 0.2, 0.1])

    assert logprob_gap_from_probs(probs) == pytest.approx(np.log(0.7) - np.log(0.2))


def test_attention_rollout_with_entropy_returns_valid_tour_and_finite_entropy():
    torch.manual_seed(5)
    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    points = np.random.default_rng(0).random((5, 2)).astype(np.float32)

    route, length, mean_entropy = attention_rollout_with_entropy(model, points)

    assert validate_tour(route, 5)
    assert length > 0
    assert np.isfinite(mean_entropy)
    assert mean_entropy >= 0


def test_policy_rollout_signals_and_stochastic_variance_are_finite():
    torch.manual_seed(5)
    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    points = np.random.default_rng(1).random((5, 2)).astype(np.float32)

    route, length, entropy, sharpness = policy_rollout_signals(model, points)
    variance, std = stochastic_length_variance(model, points, samples=3)

    assert validate_tour(route, 5)
    assert length > 0
    assert entropy >= 0
    assert sharpness >= 0
    assert variance >= 0
    assert std >= 0


def test_permutation_experiment_writes_expected_rows_for_attention_model():
    torch.manual_seed(5)
    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)

    raw, summary = run_permutation_experiment(
        [{"method": "Attention:test", "model_object": model, "decoder": "greedy"}],
        n_cities=5,
        num_instances=2,
        num_permutations=3,
        scale=10.0,
        seed=3,
    )

    assert len(raw) == 6
    assert set(summary["method"]) == {"Attention:test"}
    assert summary["count"].iloc[0] == 6
    assert raw["hamming_distance"].between(0, 1).all()


def test_diagnostic_summaries_include_confidence_intervals():
    torch.manual_seed(6)
    model = AttentionModel(d_model=16, n_heads=4, num_layers=1)
    entries = [{"method": "Attention:test", "model_object": model, "decoder": "greedy"}]

    _, permutation_summary = run_permutation_experiment(
        entries,
        n_cities=5,
        num_instances=2,
        num_permutations=2,
        scale=1.0,
        seed=22,
    )
    _, noise_summary = run_noise_robustness(
        entries,
        n_cities=5,
        num_instances=2,
        sigmas=[0.0, 0.1],
        scale=1.0,
        seed=23,
        exact_max_n=5,
        reference_solver="exact_dp",
        lkh_scale=1_000_000,
    )
    _, entropy_summary = run_entropy_experiment(
        entries,
        n_cities_list=[5],
        num_instances=2,
        scale=1.0,
        seed=24,
        exact_max_n=5,
        reference_solver="exact_dp",
        lkh_scale=1_000_000,
    )

    assert "length_abs_diff_ci95" in permutation_summary.columns
    assert "hamming_ci95" in permutation_summary.columns
    assert "gap_ci95" in noise_summary.columns
    assert "gap_ci95" in entropy_summary.columns
    assert "entropy_ci95" in entropy_summary.columns
    assert "logprob_gap_ci95" in entropy_summary.columns
    assert "sample_length_variance_ci95" in entropy_summary.columns
    assert "entropy_pearson_r" in entropy_summary.columns
    assert "logprob_gap_pearson_r" in entropy_summary.columns
    assert "sample_variance_pearson_r" in entropy_summary.columns


def test_baseline_advantages_distinguish_current_and_true_greedy_rollout():
    sampled = torch.tensor([12.0, 18.0, 30.0])
    greedy = torch.tensor([10.0, 20.0, 40.0])

    current_advantage = BatchGreedyMeanBaseline().advantages(sampled, greedy)
    rollout_advantage = GreedyRolloutBaseline().advantages(sampled, greedy)

    assert torch.allclose(current_advantage, torch.tensor([-11.333333, -5.333333, 6.666667]), atol=1e-5)
    assert torch.allclose(rollout_advantage, torch.tensor([2.0, -2.0, -10.0]))


def test_attention_model_can_force_pomo_starts():
    torch.manual_seed(7)
    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    coords = torch.rand(1, 5, 2)
    expanded, starts = expand_pomo_batch(coords, POMOConfig(num_starts=5))

    actions, lengths, log_probs = model(expanded, decode_mode="sample", fixed_start=starts)

    assert actions.shape == (5, 5)
    assert lengths.shape == (5,)
    assert log_probs.shape == (5, 4)
    assert torch.equal(actions[:, 0].cpu(), torch.arange(5))
    for row in actions.cpu().numpy():
        assert validate_tour(row, 5)


@pytest.mark.parametrize(
    "model",
    [
        AttentionModel(d_model=32, n_heads=4, num_layers=1),
        PointerNet(hidden_dim=16),
    ],
)
def test_trainable_architectures_sample_and_greedy_decode_valid_tours(model):
    torch.manual_seed(11)
    coords = torch.rand(2, 5, 2)

    sampled_actions, sampled_lengths, sampled_log_probs = model(coords, decode_mode="sample")
    greedy_actions, greedy_lengths, greedy_log_probs = model(coords, decode_mode="greedy")

    assert sampled_actions.shape == (2, 5)
    assert sampled_lengths.shape == (2,)
    assert sampled_log_probs.shape == (2, 5)
    assert greedy_actions.shape == (2, 5)
    assert greedy_lengths.shape == (2,)
    assert greedy_log_probs is None
    for row in torch.cat([sampled_actions, greedy_actions], dim=0).cpu().numpy():
        assert validate_tour(row, 5)


@pytest.mark.parametrize(
    "model",
    [
        AttentionModel(d_model=32, n_heads=4, num_layers=1),
        PointerNet(hidden_dim=16),
    ],
)
def test_trainable_architectures_reject_invalid_fixed_starts(model):
    coords = torch.rand(2, 5, 2)

    with pytest.raises(ValueError, match="fixed_start"):
        model(coords, decode_mode="sample", fixed_start=torch.tensor([0, 5]))

    with pytest.raises(ValueError, match="fixed_start"):
        model(coords, decode_mode="sample", fixed_start=torch.tensor([0, -1]))


def test_pomo_group_advantage_is_centered_per_instance():
    lengths = torch.tensor([10.0, 12.0, 14.0, 5.0, 7.0, 9.0])
    advantage = pomo_group_advantage(lengths, batch_size=2, num_starts=3)

    assert torch.allclose(advantage.view(2, 3).mean(dim=1), torch.zeros(2))
    assert torch.allclose(advantage, torch.tensor([-2.0, 0.0, 2.0, -2.0, 0.0, 2.0]))


def test_pomo_decode_preserves_training_mode():
    from rl4tsp.pomo import pomo_decode_best

    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    model.train()
    coords = torch.rand(2, 5, 2)

    pomo_decode_best(model, coords, num_starts=3)

    assert model.training


def test_pomo_decode_returns_best_fixed_start_rollout():
    from torch import nn

    from rl4tsp.pomo import pomo_decode_best

    class FixedStartLengthModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, coords, decode_mode="greedy", fixed_start=None):
            batch_size, n_cities, _ = coords.shape
            assert fixed_start is not None
            base = torch.arange(n_cities, device=coords.device).unsqueeze(0).repeat(batch_size, 1)
            actions = torch.roll(base, shifts=-1, dims=1)
            actions[:, 0] = fixed_start
            for row, start in enumerate(fixed_start.tolist()):
                remaining = [node for node in range(n_cities) if node != start]
                actions[row] = torch.tensor([start, *remaining], device=coords.device)
            lengths = (10 - fixed_start.float()) + self.weight
            return actions, lengths, None

    coords = torch.rand(1, 4, 2)
    actions, lengths = pomo_decode_best(FixedStartLengthModel(), coords, num_starts=4)

    assert actions.tolist() == [[3, 0, 1, 2]]
    assert torch.allclose(lengths, torch.tensor([7.0]))


def test_pomo_config_rejects_nonpositive_num_starts():
    with pytest.raises(ValueError, match="num_starts"):
        expand_pomo_batch(torch.rand(1, 5, 2), POMOConfig(num_starts=0))

    with pytest.raises(ValueError, match="num_starts"):
        expand_pomo_batch(torch.rand(1, 5, 2), POMOConfig(num_starts=-1))


def test_pomo_public_helpers_reject_nonpositive_num_starts():
    from rl4tsp.pomo import pomo_decode_best, pomo_training_loss

    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    coords = torch.rand(1, 5, 2)

    with pytest.raises(ValueError, match="num_starts"):
        pomo_decode_best(model, coords, num_starts=0)

    with pytest.raises(ValueError, match="num_starts"):
        pomo_training_loss(model, coords, num_starts=0)


def test_reinforce_greedy_baseline_uses_eval_mode_then_restores_training():
    from torch import nn

    from rl4tsp.train import train_reinforce

    class ModeRecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.calls: list[tuple[str, bool]] = []

        def forward(self, coords, decode_mode="sample"):
            self.calls.append((decode_mode, self.training))
            batch_size, n_cities, _ = coords.shape
            actions = torch.zeros(batch_size, n_cities, dtype=torch.long, device=coords.device)
            if decode_mode == "sample":
                lengths = torch.ones(batch_size, device=coords.device)
                log_probs = self.weight.expand(batch_size, 1)
                return actions, lengths, log_probs
            return actions, torch.zeros(batch_size, device=coords.device), None

    model = ModeRecordingModel()

    train_reinforce(
        model,
        seed=1,
        n_train=5,
        batch_size=2,
        epochs=1,
        lr=0.01,
        baseline_name="greedy_rollout",
    )

    assert model.calls == [("sample", True), ("greedy", False)]


def test_reinforce_training_restores_best_observed_weights():
    from torch import nn

    from rl4tsp.train import train_reinforce

    class BestStateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, coords, decode_mode="sample"):
            batch_size, n_cities, _ = coords.shape
            actions = torch.zeros(batch_size, n_cities, dtype=torch.long, device=coords.device)
            if decode_mode == "sample":
                lengths = torch.ones(batch_size, device=coords.device)
                log_probs = self.weight.expand(batch_size, 1)
                return actions, lengths, log_probs
            lengths = self.weight.square().expand(batch_size)
            return actions, lengths, None

    trained, log = train_reinforce(
        BestStateModel(),
        seed=1,
        n_train=5,
        batch_size=2,
        epochs=2,
        lr=0.1,
        baseline_name="greedy_rollout",
    )

    assert trained.weight.item() == pytest.approx(0.0)
    assert log[0]["is_best"] is True
    assert log[-1]["best_epoch"] == 1
    assert log[-1]["best_metric_name"] == "greedy_mean_length"


def test_pomo_training_uses_periodic_greedy_evaluation():
    from torch import nn

    from rl4tsp.train import train_pomo

    class GreedyRecordingPomoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.greedy_calls = 0

        def forward(self, coords, decode_mode="sample", fixed_start=None):
            batch_size, n_cities, _ = coords.shape
            if fixed_start is None:
                fixed_start = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
            actions = torch.arange(n_cities, device=coords.device).unsqueeze(0).repeat(batch_size, 1)
            actions[:, 0] = fixed_start
            lengths = fixed_start.float() + 1.0
            if decode_mode == "sample":
                log_probs = self.weight.expand(batch_size, 1)
                return actions, lengths, log_probs
            self.greedy_calls += 1
            return actions, lengths, None

    model = GreedyRecordingPomoModel()
    _, log = train_pomo(
        model,
        seed=1,
        n_train=4,
        batch_size=2,
        epochs=5,
        lr=0.1,
        num_starts=2,
        eval_interval=3,
    )

    assert model.greedy_calls == 3
    assert [entry["pomo_greedy_best_mean_length"] is not None for entry in log] == [True, False, True, False, True]
    assert log[-1]["best_metric_name"] == "pomo_sample_best_length"


def test_artifact_writer_saves_raw_summary_config_and_metadata(tmp_path: Path):
    raw = pd.DataFrame(
        [
            {"method": "Attention", "n_cities": 5, "length": 10.0, "time_ms": 1.0},
            {"method": "POMO", "n_cities": 5, "length": 9.0, "time_ms": 1.5},
        ]
    )
    summary = build_summary(raw, group_cols=["method", "n_cities"], value_cols=["length", "time_ms"])

    outputs = save_experiment_outputs(
        root=tmp_path,
        experiment_name="smoke",
        config={"seed": 1},
        raw=raw,
        summary=summary,
        metadata={"note": "test"},
    )

    assert outputs.raw_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.config_json.exists()
    assert outputs.metadata_json.exists()
    loaded_summary = pd.read_csv(outputs.summary_csv)
    assert set(loaded_summary["method"]) == {"Attention", "POMO"}


def test_build_summary_reports_standard_error_and_95_percent_ci():
    from scipy import stats

    raw = pd.DataFrame(
        [
            {"method": "A", "n_cities": 5, "gap_percent": 1.0, "time_ms": 10.0},
            {"method": "A", "n_cities": 5, "gap_percent": 2.0, "time_ms": 12.0},
            {"method": "A", "n_cities": 5, "gap_percent": 3.0, "time_ms": 14.0},
            {"method": "B", "n_cities": 5, "gap_percent": 10.0, "time_ms": 20.0},
        ]
    )

    summary = build_summary(raw, group_cols=["method", "n_cities"], value_cols=["gap_percent", "time_ms"])
    row_a = summary[summary["method"] == "A"].iloc[0]
    row_b = summary[summary["method"] == "B"].iloc[0]

    assert row_a["gap_percent_mean"] == pytest.approx(2.0)
    assert row_a["gap_percent_std"] == pytest.approx(1.0)
    assert row_a["count"] == 3
    assert row_a["gap_percent_sem"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert row_a["gap_percent_ci95"] == pytest.approx(stats.t.ppf(0.975, df=2) / np.sqrt(3.0))
    assert row_a["time_ms_sem"] == pytest.approx(2.0 / np.sqrt(3.0))
    assert row_a["time_ms_ci95"] == pytest.approx(stats.t.ppf(0.975, df=2) * 2.0 / np.sqrt(3.0))
    assert row_b["gap_percent_std"] == 0.0
    assert row_b["gap_percent_sem"] == 0.0
    assert row_b["gap_percent_ci95"] == 0.0


def test_plot_tsp_summary_prefers_ci95_error_bars(tmp_path: Path, monkeypatch):
    summary = pd.DataFrame(
        [
            {
                "method": "A",
                "n_cities": 5,
                "gap_percent_mean": 2.0,
                "gap_percent_std": 10.0,
                "gap_percent_ci95": 0.5,
                "time_ms_mean": 12.0,
                "time_ms_std": 20.0,
                "time_ms_ci95": 1.5,
            }
        ]
    )
    summary_csv = tmp_path / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    yerrs = []
    bands = []

    class FakeFigure:
        def tight_layout(self):
            pass

        def savefig(self, path, dpi):
            Path(path).write_bytes(b"fake image")

    class FakeAxes:
        def fill_between(self, x, y1, y2, **kwargs):
            bands.append((list(y1), list(y2)))

        def errorbar(self, x, y, yerr=None, **kwargs):
            yerrs.append(list(yerr) if yerr is not None else None)

        def set_xlabel(self, label):
            pass

        def set_ylabel(self, label):
            pass

        def set_title(self, label):
            pass

        def set_yscale(self, scale):
            pass

        def grid(self, *args, **kwargs):
            pass

        def legend(self):
            pass

    monkeypatch.setattr(plots.plt, "subplots", lambda figsize: (FakeFigure(), FakeAxes()))
    monkeypatch.setattr(plots.plt, "close", lambda fig: None)

    written = plots.plot_tsp_summary(summary_csv, tmp_path / "figures")

    assert [path.name for path in written] == ["tsp_gap.png", "tsp_time.png"]
    assert yerrs == [[0.5], [1.5]]
    assert bands == []


def test_diagnostic_plots_prefer_ci95_error_bars(tmp_path: Path, monkeypatch):
    from rl4tsp import diagnostics

    calls: list[tuple[str, list[float]]] = []
    bands: list[tuple[list[float], list[float]]] = []

    class FakeFigure:
        def tight_layout(self):
            pass

        def savefig(self, path, dpi):
            Path(path).write_bytes(b"fake image")

    class FakeAxis:
        def fill_between(self, x, y1, y2, **kwargs):
            bands.append((list(y1), list(y2)))

        def bar(self, x, height, yerr=None, **kwargs):
            calls.append(("bar", list(yerr)))

        def errorbar(self, x, y, yerr=None, **kwargs):
            calls.append(("errorbar", list(yerr)))

        def set_xlabel(self, label):
            pass

        def set_ylabel(self, label):
            pass

        def set_title(self, label):
            pass

        def tick_params(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

    monkeypatch.setattr(diagnostics.plt, "subplots", lambda *args, **kwargs: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr(diagnostics.plt, "close", lambda fig: None)

    plot_permutation_summary(
        pd.DataFrame(
            [
                {"method": "A", "hamming_mean": 0.2, "hamming_std": 10.0, "hamming_ci95": 0.03},
            ]
        ),
        tmp_path / "permutation.png",
    )
    plot_noise_summary(
        pd.DataFrame(
            [
                {"method": "A", "noise_percent": 0.0, "gap_mean": 1.0, "gap_std": 10.0, "gap_ci95": 0.04},
            ]
        ),
        tmp_path / "noise.png",
    )

    assert calls == [("bar", [0.03]), ("errorbar", [0.04])]
    assert bands == [([0.96], [1.04])]


def test_diagnostics_outputs_write_all_pngs_to_figures_dir(tmp_path: Path):
    experiment_dir = tmp_path / "tsp_diagnostics"
    figures_dir = tmp_path / "figures"

    written = save_diagnostics_outputs(
        output_dir=experiment_dir,
        figures_dir=figures_dir,
        config={"seed": 1},
        permutation_raw=pd.DataFrame([{"method": "A", "hamming_distance": 0.2}]),
        permutation_summary=pd.DataFrame(
            [{"method": "A", "hamming_mean": 0.2, "hamming_ci95": 0.03}]
        ),
        noise_raw=pd.DataFrame([{"method": "A", "gap_percent": 1.0}]),
        noise_summary=pd.DataFrame(
            [{"method": "A", "noise_percent": 0.0, "gap_mean": 1.0, "gap_ci95": 0.04}]
        ),
        entropy_raw=pd.DataFrame(
            [
                {
                    "method": "A",
                    "mean_entropy": 0.5,
                    "logprob_gap": 0.1,
                    "sample_length_variance": 0.2,
                    "gap_percent": 1.0,
                }
            ]
        ),
        entropy_summary=pd.DataFrame(
            [
                {
                    "method": "A",
                    "n_cities": 5,
                    "entropy_pearson_r": 0.1,
                    "entropy_r_ci_low": -0.1,
                    "entropy_r_ci_high": 0.3,
                    "logprob_gap_pearson_r": 0.2,
                    "logprob_gap_r_ci_low": 0.0,
                    "logprob_gap_r_ci_high": 0.4,
                    "sample_variance_pearson_r": 0.3,
                    "sample_variance_r_ci_low": 0.1,
                    "sample_variance_r_ci_high": 0.5,
                }
            ]
        ),
    )

    for filename in [
        "permutation_invariance.png",
        "noise_robustness.png",
        "entropy_gap.png",
        "entropy_correlations.png",
    ]:
        assert (figures_dir / filename).exists()
        assert not (experiment_dir / filename).exists()
    assert Path(written["permutation_plot"]).parent == figures_dir


def test_tsp_plot_module_uses_noninteractive_backend():
    assert plots.plt.get_backend().lower() == "agg"


def test_compare_loader_rejects_missing_checkpoint(tmp_path: Path):
    from experiments.run_tsp_compare import _load_required_checkpoint

    with pytest.raises(FileNotFoundError, match="configured checkpoint"):
        _load_required_checkpoint(str(tmp_path / "missing.pt"), "attention", hidden_dim=32)


def test_build_attention_model_accepts_nonstandard_hidden_dim():
    from rl4tsp.train import build_model

    model = build_model("attention", hidden_dim=31)
    coords = torch.rand(1, 5, 2)
    actions, lengths, _ = model(coords, decode_mode="greedy")

    assert actions.shape == (1, 5)
    assert lengths.shape == (1,)


def test_build_attention_model_uses_original_three_encoder_layers():
    from rl4tsp.train import build_model

    model = build_model("attention", hidden_dim=32)

    assert len(model.encoder.layers) == 3


def test_greedy_benchmark_decode_preserves_training_mode():
    from rl4tsp.benchmark import decode_greedy

    model = AttentionModel(d_model=32, n_heads=4, num_layers=1)
    model.train()
    points = np.random.default_rng(0).random((5, 2)).astype(np.float32)

    decode_greedy(model, points)

    assert model.training


def test_small_compare_config_includes_all_trained_reinforce_variants_and_pomo():
    config = json.loads(Path("configs/tsp_compare_small.json").read_text(encoding="utf-8"))
    methods = {entry["method"] for entry in config["model_checkpoints"]}

    assert {
        "Attention:batch_greedy_mean",
        "Attention:greedy_rollout",
        "PointerNet:batch_greedy_mean",
        "PointerNet:greedy_rollout",
    }.issubset(methods)
    assert config["pomo_checkpoint"]
