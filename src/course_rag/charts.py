"""Charts for the README (matplotlib, optional: pip install ".[report]")."""

from __future__ import annotations

from pathlib import Path

from .evals import SweepResult

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-3 (validated for all pairs, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK_2)


def plot_sweep(sweep: SweepResult, previous: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    i_best = sweep.thresholds.index(min(sweep.thresholds, key=lambda t: abs(t - sweep.best)))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(
        sweep.thresholds,
        [v * 100 for v in sweep.answer_rate],
        color=SERIES[0],
        linewidth=2,
        label="Answerable questions answered",
    )
    ax.plot(
        sweep.thresholds,
        [v * 100 for v in sweep.refusal_rate],
        color=SERIES[1],
        linewidth=2,
        label="Unanswerable questions refused",
    )
    ax.axvline(sweep.best, color=INK, linewidth=1, linestyle="--")
    ax.axvline(previous, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(
        f"chosen {sweep.best:.3f}",
        (sweep.best, 8),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax.annotate(
        f"old guess {previous:.2f}",
        (previous, 20),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
        ha="right",
    )
    _style(
        ax,
        f"A threshold of {sweep.best:.3f} answers {sweep.answer_rate[i_best]:.0%} of course "
        f"questions and refuses {sweep.refusal_rate[i_best]:.0%} of off-topic ones",
        "Refusal threshold on retrieval relevance (hybrid setup), share of questions in %. "
        "42-question eval set.",
    )
    ax.set_xlim(0, max(sweep.thresholds))
    ax.set_ylim(0, 105)
    _legend(ax, loc="upper left", ncol=2, bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_mrr_by_kind(by_setup: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    setups = list(by_setup)
    kinds = sorted(next(iter(by_setup.values())))
    x = np.arange(len(kinds))
    width = 0.8 / len(setups)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    for i, setup in enumerate(setups):
        values = [by_setup[setup][k]["mrr"] for k in kinds]
        bars = ax.bar(
            x + (i - (len(setups) - 1) / 2) * width,
            values,
            width * 0.92,
            color=SERIES[i],
            label=setup,
            edgecolor=SURFACE,
            linewidth=1,
        )
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    para = {s: by_setup[s].get("paraphrase", {}).get("mrr", 0) for s in setups}
    winner = max(para, key=lambda s: para[s])
    _style(
        ax,
        f"Paraphrased questions are where retrievers differ; {winner} ranks best "
        f"(MRR {para[winner]:.2f})",
        "Mean reciprocal rank of the first relevant chunk, by question type. "
        "Offline setup (hashing embeddings).",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} (n={by_setup[setups[0]][k]['n']:.0f})" for k in kinds])
    ax.set_ylim(0, 1.12)
    _legend(ax, loc="upper left", ncol=len(setups), bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
