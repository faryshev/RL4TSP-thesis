from pathlib import Path


def test_package_is_tsp_only():
    forbidden_name = "meta" + "poc"
    forbidden = [
        Path("src") / forbidden_name,
        Path("experiments") / f"run_{forbidden_name}_gp.py",
        Path("experiments") / f"run_{forbidden_name}_{'oracle' + '_choice'}.py",
        Path("experiments") / f"run_{forbidden_name}_multifidelity.py",
        Path("configs") / f"{forbidden_name}_small.json",
    ]

    assert [path for path in forbidden if path.exists()] == []


def test_required_tsp_entrypoints_exist():
    required = [
        Path("experiments/run_tsp_reinforce.py"),
        Path("experiments/run_tsp_pomo.py"),
        Path("experiments/run_tsp_compare.py"),
        Path("experiments/run_tsp_diagnostics.py"),
        Path("experiments/run_tsp_reward_noise.py"),
        Path("Experiment_Launcher_and_Analysis.ipynb"),
        Path("run_experiments.sh"),
    ]

    assert [path for path in required if not path.exists()] == []


def test_required_full_configs_exist():
    required = [
        Path("configs/tsp_reinforce_full.json"),
        Path("configs/tsp_pomo_full.json"),
        Path("configs/tsp_compare_full.json"),
        Path("configs/tsp_diagnostics_full.json"),
        Path("configs/tsp_reward_noise_full.json"),
    ]

    assert [path for path in required if not path.exists()] == []
