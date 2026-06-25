from __future__ import annotations

from pathlib import Path
from typing import Any


PALETTE = {
    "background": "#FDFFFC",
    "text": "#2F312E",
    "grid": "#D0D0D0",
    "reinforce": "#7785AC",
    "pointer": "#6A8D73",
    "attention": "#B75E2F",
    "lkh": "#2F312E",
    "baseline": "#7F7F7F",
    "pomo": "#4C78A8",
    "signal": "#A05A7B",
}

CI95_NOTE = "Точки/столбцы показывают среднее; вертикальные интервалы показывают 95% доверительный интервал."


METHOD_STYLES = {
    "LKH": {"color": PALETTE["lkh"], "marker": "o", "linestyle": "-"},
    "Reference:exact_dp": {"color": PALETTE["lkh"], "marker": "o", "linestyle": "-"},
    "Reference:nn_2opt": {"color": PALETTE["lkh"], "marker": "o", "linestyle": "-"},
    "PointerNet": {"color": PALETTE["pointer"], "marker": "s", "linestyle": "--"},
    "PointerNet:batch_greedy_mean": {"color": PALETTE["pointer"], "marker": "s", "linestyle": "-."},
    "PointerNet:greedy_rollout": {"color": "#275D38", "marker": "D", "linestyle": ":"},
    "pointer:batch_greedy_mean": {"color": PALETTE["pointer"], "marker": "s", "linestyle": "-."},
    "pointer:greedy_rollout": {"color": "#275D38", "marker": "D", "linestyle": ":"},
    "Attention": {"color": PALETTE["attention"], "marker": "^", "linestyle": "-"},
    "Attention:batch_greedy_mean": {"color": PALETTE["attention"], "marker": "^", "linestyle": "-"},
    "Attention:greedy_rollout": {"color": "#8C2D18", "marker": "v", "linestyle": "--"},
    "attention:batch_greedy_mean": {"color": PALETTE["attention"], "marker": "^", "linestyle": "-"},
    "attention:greedy_rollout": {"color": "#8C2D18", "marker": "v", "linestyle": "--"},
    "POMO": {"color": PALETTE["pomo"], "marker": "P", "linestyle": (0, (5, 2, 1, 2))},
    "pomo": {"color": PALETTE["pomo"], "marker": "P", "linestyle": (0, (5, 2, 1, 2))},
}


def style_for(name: str) -> dict[str, Any]:
    return METHOD_STYLES.get(name, {"color": PALETTE["baseline"], "marker": "o", "linestyle": "-"})


def apply_figure_style(fig: Any) -> None:
    patch = getattr(fig, "patch", None)
    if patch is not None and hasattr(patch, "set_facecolor"):
        patch.set_facecolor(PALETTE["background"])


def style_axis(ax: Any, *, grid_axis: str | None = None) -> None:
    if hasattr(ax, "set_facecolor"):
        ax.set_facecolor(PALETTE["background"])
    if hasattr(ax, "grid"):
        kwargs = {"linestyle": ":", "color": PALETTE["grid"], "alpha": 0.8}
        if grid_axis is None:
            ax.grid(True, **kwargs)
        else:
            ax.grid(True, axis=grid_axis, **kwargs)
    if hasattr(ax, "tick_params"):
        ax.tick_params(colors=PALETTE["text"])
    spines = getattr(ax, "spines", {})
    for side in ("top", "right"):
        spine = spines.get(side) if hasattr(spines, "get") else None
        if spine is not None and hasattr(spine, "set_visible"):
            spine.set_visible(False)
    for side in ("left", "bottom"):
        spine = spines.get(side) if hasattr(spines, "get") else None
        if spine is not None and hasattr(spine, "set_color"):
            spine.set_color(PALETTE["text"])
    for artist in (getattr(ax, "title", None), getattr(ax, "xaxis", None), getattr(ax, "yaxis", None)):
        if artist is None:
            continue
        label = artist if hasattr(artist, "set_color") else getattr(artist, "label", None)
        if label is not None and hasattr(label, "set_color"):
            label.set_color(PALETTE["text"])
    legend = ax.get_legend() if hasattr(ax, "get_legend") else None
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(PALETTE["background"])
        frame.set_edgecolor(PALETTE["grid"])
        for text in legend.get_texts():
            text.set_color(PALETTE["text"])


def add_ci95_note(ax: Any, text: str = CI95_NOTE) -> None:
    if not hasattr(ax, "text") or not hasattr(ax, "transAxes"):
        return
    ax.text(
        0.0,
        -0.22,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color=PALETTE["text"],
        va="top",
    )


def save_figure(fig: Any, path: Path, *, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["background"])
    return path
