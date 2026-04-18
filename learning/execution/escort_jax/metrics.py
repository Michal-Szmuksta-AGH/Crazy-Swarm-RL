from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def save_metrics_csv(metrics_path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep all metric columns so newly added reward terms are persisted to CSV.
    fieldnames = list(rows[0].keys())
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(plot_path: Path, rows: List[dict]) -> None:
    if not rows:
        return

    updates = np.array([row["update"] for row in rows], dtype=np.int32)
    returns = np.array([row["return"] for row in rows], dtype=np.float32)
    raw_returns = np.array([row["raw_return"] for row in rows], dtype=np.float32)
    rolling = np.array([row["rolling_return"] for row in rows], dtype=np.float32)
    raw_rolling = np.array([row["raw_rolling_return"] for row in rows], dtype=np.float32)
    actor_losses = np.array([row["actor_loss"] for row in rows], dtype=np.float32)
    critic_losses = np.array([row["critic_loss"] for row in rows], dtype=np.float32)
    entropies = np.array([row["entropy"] for row in rows], dtype=np.float32)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(updates, returns, alpha=0.25, label="normalized return")
    axes[0, 0].plot(updates, rolling, linewidth=2.0, label="normalized rolling")
    ax_raw = axes[0, 0].twinx()
    ax_raw.plot(updates, raw_returns, alpha=0.20, color="tab:green", label="raw return")
    ax_raw.plot(updates, raw_rolling, linewidth=1.8, color="tab:green", linestyle="--", label="raw rolling")
    axes[0, 0].set_title("Return")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = ax_raw.get_legend_handles_labels()
    axes[0, 0].legend(h1 + h2, l1 + l2, loc="best")
    ax_raw.grid(False)

    axes[0, 1].plot(updates, actor_losses)
    axes[0, 1].set_title("Actor loss")

    axes[1, 0].plot(updates, critic_losses)
    axes[1, 0].set_title("Critic loss")

    axes[1, 1].plot(updates, entropies)
    axes[1, 1].set_title("Entropy")

    for ax in axes.ravel():
        ax.set_xlabel("update")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
