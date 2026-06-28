import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from rl4tsp.reward_noise import (
    apply_relative_reward_noise,
    compute_noisy_advantage,
    evaluate_model_true,
    run_reward_noise_experiment,
    train_pomo_with_reward_noise,
    train_reinforce_with_reward_noise,
)


class FixedLengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, coords, decode_mode="sample"):
        batch_size, n_cities, _ = coords.shape
        actions = torch.arange(n_cities, device=coords.device).unsqueeze(0).repeat(batch_size, 1)
        base = torch.arange(1, batch_size + 1, device=coords.device, dtype=torch.float32)
        if decode_mode == "sample":
            lengths = 10.0 + base + self.weight * 0.0
            log_probs = self.weight.expand(batch_size, 1)
            return actions, lengths, log_probs
        lengths = 8.0 + base + self.weight * 0.0
        return actions, lengths, None


class BestStateRewardNoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, coords, decode_mode="sample"):
        batch_size, n_cities, _ = coords.shape
        actions = torch.arange(n_cities, device=coords.device).unsqueeze(0).repeat(batch_size, 1)
        if decode_mode == "sample":
            lengths = 1.0 + self.weight.square().expand(batch_size)
            log_probs = self.weight.expand(batch_size, 1)
            return actions, lengths, log_probs
        lengths = torch.zeros(batch_size, device=coords.device)
        return actions, lengths, None


def test_sigma_zero_observed_lengths_equal_true_lengths():
    true_lengths = torch.tensor([1.0, 2.0, 3.0])

    observed = apply_relative_reward_noise(true_lengths, sigma_rel=0.0)

    assert torch.allclose(observed, true_lengths)


def test_nonzero_sigma_changes_observed_lengths_without_mutating_true_lengths():
    true_lengths = torch.full((32,), 10.0)
    before = true_lengths.clone()
    generator = torch.Generator().manual_seed(7)

    observed = apply_relative_reward_noise(true_lengths, sigma_rel=0.2, generator=generator)

    assert not torch.allclose(observed, true_lengths)
    assert torch.allclose(true_lengths, before)


def test_greedy_rollout_noises_sample_and_baseline_independently():
    sampled_true = torch.full((16,), 10.0)
    greedy_true = torch.full((16,), 10.0)
    generator = torch.Generator().manual_seed(11)

    result = compute_noisy_advantage(
        sampled_true,
        greedy_true,
        sigma_rel=0.4,
        baseline_name="greedy_rollout",
        generator=generator,
    )

    assert not torch.allclose(result.sampled_observed, sampled_true)
    assert not torch.allclose(result.baseline_observed, greedy_true)
    assert not torch.allclose(result.sampled_observed, result.baseline_observed)
    assert torch.allclose(result.advantage, result.sampled_observed - result.baseline_observed)


def test_batch_mean_baseline_uses_noisy_baseline_mean():
    sampled_true = torch.full((8,), 10.0)
    greedy_true = torch.full((8,), 12.0)
    generator = torch.Generator().manual_seed(13)

    result = compute_noisy_advantage(
        sampled_true,
        greedy_true,
        sigma_rel=0.3,
        baseline_name="batch_greedy_mean",
        generator=generator,
    )

    assert torch.allclose(result.advantage, result.sampled_observed - result.baseline_observed.mean())


def test_training_log_records_noisy_objectives_separately_from_true_lengths():
    model = FixedLengthModel()

    _, log = train_reinforce_with_reward_noise(
        model,
        seed=5,
        n_train=2,
        train_size=5,
        batch_size=2,
        epochs=1,
        lr=0.01,
        baseline_name="greedy_rollout",
        sigma_rel=0.0,
    )

    assert len(log) == 1
    assert log[0]["observed_length_mean"] == pytest.approx(log[0]["true_length_mean"])
    assert log[0]["baseline_observed_mean"] == pytest.approx(log[0]["baseline_true_mean"])


def test_reward_noise_training_restores_best_observed_weights():
    trained, log = train_reinforce_with_reward_noise(
        BestStateRewardNoiseModel(),
        seed=5,
        n_train=2,
        train_size=5,
        batch_size=2,
        epochs=2,
        lr=0.1,
        baseline_name="greedy_rollout",
        sigma_rel=0.0,
    )

    assert trained.weight.item() == pytest.approx(0.0)
    assert log[0]["is_best"] is True
    assert log[-1]["best_epoch"] == 1
    assert log[-1]["best_metric_name"] == "true_length_mean"


def test_pomo_reward_noise_training_runs_and_logs_best_state():
    torch.manual_seed(7)
    from rl4tsp.models import AttentionModel

    trained, log = train_pomo_with_reward_noise(
        AttentionModel(d_model=16, n_heads=4, num_layers=1),
        seed=7,
        n_train=2,
        train_size=4,
        batch_size=1,
        epochs=1,
        lr=0.0001,
        num_starts=4,
        sigma_rel=0.0,
    )

    assert trained.training is False
    assert len(log) == 1
    assert log[0]["best_metric_name"] == "pomo_true_best_mean_length"


def test_evaluation_uses_true_lengths_for_all_sigma_values():
    model = FixedLengthModel()
    config = {
        "seed": 3,
        "scale": 1.0,
        "eval_sizes": [4],
        "eval_instances": 2,
        "exact_max_n": 4,
        "reference_solver": "exact_dp",
        "lkh_scale": 1000000,
    }

    zero_rows = evaluate_model_true(model, config=config, sigma_rel=0.0, epoch=1, method="fixed")
    noisy_rows = evaluate_model_true(model, config=config, sigma_rel=0.7, epoch=1, method="fixed")

    zero = pd.DataFrame(zero_rows).sort_values(["n_cities", "instance"]).reset_index(drop=True)
    noisy = pd.DataFrame(noisy_rows).sort_values(["n_cities", "instance"]).reset_index(drop=True)
    assert zero["length"].tolist() == noisy["length"].tolist()
    assert zero["gap_percent"].tolist() == noisy["gap_percent"].tolist()


def test_reward_noise_experiment_writes_expected_files(tmp_path: Path):
    config = {
        "seed": 19,
        "output_root": str(tmp_path),
        "experiment_name": "reward_noise_test",
        "scale": 1.0,
        "device": "cpu",
        "model": "attention",
        "models": ["attention", "pointer", "pomo"],
        "n_train": 2,
        "train_size": 4,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 0.0001,
        "hidden_dim": 16,
        "baseline": "batch_greedy_mean",
        "baselines": ["greedy_rollout"],
        "pomo_num_starts": 4,
        "sigma_rel_values": [0.0],
        "eval_sizes": [4],
        "eval_instances": 1,
        "exact_max_n": 4,
        "reference_solver": "exact_dp",
        "lkh_scale": 1000000,
    }

    raw, summary = run_reward_noise_experiment(config)

    experiment_dir = tmp_path / "reward_noise_test"
    assert (experiment_dir / "raw.csv").exists()
    assert (experiment_dir / "summary.csv").exists()
    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "metadata.json").exists()
    assert (tmp_path / "figures" / "reward_noise_gap.png").exists()
    assert (tmp_path / "figures" / "reward_noise_training.png").exists()
    assert not (experiment_dir / "reward_noise_gap.png").exists()
    assert not (experiment_dir / "reward_noise_training.png").exists()
    assert len(list((experiment_dir / "models").glob("*.pt"))) == 3
    assert {"sigma_rel", "epoch", "observed_length_mean", "true_length_mean", "baseline_observed_mean", "gap_percent", "method"}.issubset(raw.columns)
    assert set(raw["phase"]) == {"train", "eval"}
    assert {"Attention:greedy_rollout", "PointerNet:greedy_rollout", "POMO"}.issubset(set(raw["method"]))
    assert not summary.empty
    assert "gap_percent_ci95" in summary.columns
    saved_config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["experiment_name"] == "reward_noise_test"
    metadata = json.loads((experiment_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["checkpoints"]) == 3
    assert metadata["plot_sources"]["reward_noise_gap.png"]["summary_csv"].endswith("summary.csv")
    assert metadata["plot_sources"]["reward_noise_training.png"]["raw_csv"].endswith("raw.csv")
    assert metadata["figures"]["reward_noise_gap.png"].endswith("figures/reward_noise_gap.png")
    assert metadata["figures"]["reward_noise_training.png"].endswith("figures/reward_noise_training.png")
