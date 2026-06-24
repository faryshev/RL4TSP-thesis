from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only without optional SciPy.
    scipy_stats = None


@dataclass(frozen=True)
class ExperimentOutputs:
    raw_csv: Path
    summary_csv: Path
    config_json: Path
    metadata_json: Path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def add_sem_ci95_columns(summary: pd.DataFrame, value_prefixes: Iterable[str], count_col: str = "count") -> pd.DataFrame:
    counts = summary[count_col]
    for col in value_prefixes:
        std_col = f"{col}_std"
        sem_col = f"{col}_sem"
        ci_col = f"{col}_ci95"
        if std_col not in summary:
            continue
        summary[sem_col] = summary[std_col].fillna(0.0) / counts.map(math.sqrt)
        if scipy_stats is None:
            critical = 1.96
        else:
            critical = scipy_stats.t.ppf(0.975, df=counts - 1)
        summary[ci_col] = (summary[sem_col] * critical).where(counts > 1, 0.0)
    return summary.fillna(0.0)


def build_summary(raw: pd.DataFrame, group_cols: list[str], value_cols: Iterable[str]) -> pd.DataFrame:
    value_cols = list(value_cols)
    aggregations = {}
    for col in value_cols:
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, "std")
    summary = raw.groupby(group_cols, as_index=False).agg(**aggregations)
    summary["count"] = raw.groupby(group_cols).size().to_numpy()
    return add_sem_ci95_columns(summary, value_cols)


def save_experiment_outputs(
    root: Path,
    experiment_name: str,
    config: dict[str, Any],
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    experiment_dir = root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    outputs = ExperimentOutputs(
        raw_csv=experiment_dir / "raw.csv",
        summary_csv=experiment_dir / "summary.csv",
        config_json=experiment_dir / "config.json",
        metadata_json=experiment_dir / "metadata.json",
    )
    raw.to_csv(outputs.raw_csv, index=False)
    summary.to_csv(outputs.summary_csv, index=False)
    save_json(outputs.config_json, config)
    save_json(
        outputs.metadata_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **(metadata or {}),
        },
    )
    return outputs
