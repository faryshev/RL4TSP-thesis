from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

_mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "rl4tsp-matplotlib"))
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl4tsp.plot_style import METHOD_STYLES, add_ci95_note, apply_figure_style, style_axis


def _error_column(summary: pd.DataFrame, metric: str) -> str | None:
    ci_col = metric.replace("_mean", "_ci95")
    if ci_col in summary:
        return ci_col
    std_col = metric.replace("_mean", "_std")
    if std_col in summary:
        return std_col
    return None


def plot_tsp_summary(summary_csv: Path, output_dir: Path) -> list[Path]:
    summary = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, ylabel, title, filename, log_y in [
        ("gap_percent_mean", "Отклонение от опорного решения, %", "Качество решения: среднее и 95% ДИ", "tsp_gap.png", False),
        ("time_ms_mean", "Время решения, мс", "Масштабирование времени: среднее и 95% ДИ", "tsp_time.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        apply_figure_style(fig)
        for method in sorted(summary["method"].unique()):
            subset = summary[summary["method"] == method].sort_values("n_cities")
            style = METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-", "color": None})
            error_col = _error_column(summary, metric)
            yerr = subset[error_col] if error_col is not None else None
            ax.errorbar(
                subset["n_cities"],
                subset[metric],
                yerr=yerr,
                label=method,
                capsize=4,
                elinewidth=1.2,
                capthick=1.2,
                **style,
            )
        ax.set_xlabel("Число городов")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.legend()
        style_axis(ax)
        #add_ci95_note(ax)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written
