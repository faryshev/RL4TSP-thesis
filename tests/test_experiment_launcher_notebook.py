import ast
import json
from pathlib import Path


def _launcher_notebook() -> Path:
    for candidate in (
        Path("Experiment_Launcher_and_Analysis.ipynb"),
        Path("code/Experiment_Launcher_and_Analysis.ipynb"),
    ):
        if candidate.exists():
            return candidate
    raise AssertionError("Experiment_Launcher_and_Analysis.ipynb not found")


def _code_cells() -> list[str]:
    notebook = json.loads(_launcher_notebook().read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_full_run_toggles_are_top_level_independent_conditionals():
    offenders: list[str] = []
    for cell_index, source in enumerate(_code_cells()):
        parseable_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%")
        )
        tree = ast.parse(parseable_source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if isinstance(node.test, ast.Name) and node.test.id.startswith("RUN_FULL_"):
                if node.col_offset != 0:
                    offenders.append(f"cell {cell_index}: {node.test.id} at indent {node.col_offset}")

    assert offenders == []
