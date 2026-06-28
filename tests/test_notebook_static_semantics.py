import json
from pathlib import Path


NOTEBOOK = Path("Experiment_Launcher_and_Analysis.ipynb")


def _all_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_launcher_notebook_is_tsp_only():
    source = _all_source().lower()

    assert ("meta" + "poc") not in source
    assert ("мат" + "ериал") not in source
    assert "run_tsp_reinforce.py" in source
    assert "run_tsp_pomo.py" in source
    assert "run_tsp_compare.py" in source
    assert "run_tsp_diagnostics.py" in source
    assert "run_tsp_reward_noise.py" in source


def test_launcher_notebook_documents_hyperparameter_choices():
    source = _all_source()

    for token in [
        "scale = 10.0",
        "n_train = 20",
        "epochs = 3000",
        "POMO batch_size = 8",
        "exact_max_n = 10",
        "entropy_stochastic_samples = 8",
    ]:
        assert token in source


def test_launcher_notebook_uses_timestamped_results():
    source = _all_source()

    assert "RUN_ID = datetime.now().strftime" in source
    assert "results\" / RUN_ID" in source
    assert "FileExistsError" in source


def test_launcher_notebook_configures_subprocess_environment():
    source = _all_source()

    assert "PYTHON = sys.executable" in source
    assert "SRC_PATH = PROJECT_ROOT / \"src\"" in source
    assert "PYTHONPATH" in source
    assert "subprocess.run(args, cwd=PROJECT_ROOT, text=True, check=True" in source


def test_launcher_notebook_keeps_methodological_sections():
    source = _all_source()

    assert "### Общая цель исследования" in source
    assert source.count("### Гипотеза") >= 5
    assert source.count("### Цель") >= 5
    assert source.count("### Фиксированные условия") >= 5
    assert source.count("### Методика") >= 5
    assert source.count("### Интерпретация возможных исходов") >= 5
    assert "entropy_spearman_correlations.png" in source
    assert "Spearman rho" in source
