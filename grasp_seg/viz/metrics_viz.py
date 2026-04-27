"""Plot per-epoch training curves from one or several ``metrics.csv`` files.

Two modes:

- :func:`figure_single_run` — comprehensive 2×3 / 3×3 panel for one run.
- :func:`figure_compare_runs` — one figure per metric, multiple runs
  overlaid (used in the report when comparing angle / multitask-RGB /
  multitask-RGB-D heads).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading + helpers
# ---------------------------------------------------------------------------

def load_metrics(csv_path: str) -> pd.DataFrame:
    """Read ``metrics.csv`` and coerce numeric columns."""
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def _plot(ax, df: pd.DataFrame, col: str, label: str, color=None, lw: float = 1.6):
    if col in df.columns and df[col].notna().any():
        ax.plot(df["epoch"], df[col], label=label, color=color, lw=lw)


# ---------------------------------------------------------------------------
# Single-run figure
# ---------------------------------------------------------------------------

def figure_single_run(
    csv_path: str,
    title: Optional[str] = None,
    smooth: int = 1,
):
    """Comprehensive panel of all available metrics for a single run."""
    df = load_metrics(csv_path)
    if smooth > 1 and len(df) > smooth:
        for col in df.columns:
            if col == "epoch":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].rolling(smooth, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_loss, ax_iou, ax_lr = axes[0]
    ax_aux, ax_gpu, ax_time = axes[1]

    # 1) Loss
    _plot(ax_loss, df, "train_loss", "train loss", color="C0")
    if "train_pos_bce" in df.columns:
        _plot(ax_loss, df, "train_pos_bce", "pos BCE", color="C1", lw=1.0)
        _plot(ax_loss, df, "train_pos_dice", "pos Dice", color="C2", lw=1.0)
    ax_loss.set_title("Лосс по эпохам")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Значение")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # 2) val IoU / Dice
    _plot(ax_iou, df, "val_miou_fg", "val mIoU (fg)", color="C0")
    _plot(ax_iou, df, "val_dice_fg", "val Dice (fg)", color="C1")
    if "val_miou_fg_ang" in df.columns:
        _plot(ax_iou, df, "val_miou_fg_ang", "val mIoU (fg, по углу)", color="C2")
        _plot(ax_iou, df, "val_dice_fg_ang", "val Dice (fg, по углу)", color="C3")
    if "val_precision_fg" in df.columns:
        _plot(ax_iou, df, "val_precision_fg", "precision (fg)", color="C4", lw=1.0)
        _plot(ax_iou, df, "val_recall_fg", "recall (fg)", color="C5", lw=1.0)
    ax_iou.set_title("Качество сегментации (val)")
    ax_iou.set_xlabel("Эпоха")
    ax_iou.set_ylabel("Метрика, [0..1]")
    ax_iou.set_ylim(0.0, 1.0)
    ax_iou.grid(alpha=0.3)
    ax_iou.legend(loc="lower right", fontsize=8)

    # 3) LR
    _plot(ax_lr, df, "lr", "learning rate", color="C0")
    ax_lr.set_title("Learning rate (poly schedule)")
    ax_lr.set_xlabel("Эпоха")
    ax_lr.set_ylabel("lr")
    ax_lr.set_yscale("log")
    ax_lr.grid(alpha=0.3, which="both")
    ax_lr.legend()

    # 4) Multitask-only auxiliaries
    plotted = False
    for col, lab, color in [
        ("train_cos", "train cos", "C0"),
        ("train_sin", "train sin", "C1"),
        ("train_width", "train width", "C2"),
        ("val_cos_mse", "val cos MSE", "C3"),
        ("val_sin_mse", "val sin MSE", "C4"),
    ]:
        if col in df.columns:
            _plot(ax_aux, df, col, lab, color=color)
            plotted = True
    if "val_ang_mae_deg" in df.columns:
        ax_aux2 = ax_aux.twinx()
        ax_aux2.plot(df["epoch"], df["val_ang_mae_deg"], color="C7",
                     ls="--", label="val ang MAE, °")
        ax_aux2.set_ylabel("val ang MAE, °", color="C7")
        ax_aux2.tick_params(axis="y", labelcolor="C7")
        ax_aux2.legend(loc="upper right", fontsize=8)
        plotted = True
    if not plotted:
        ax_aux.text(0.5, 0.5, "Нет multitask-метрик", transform=ax_aux.transAxes,
                    ha="center", va="center", color="0.5")
    ax_aux.set_title("Multitask: cos / sin / width / ang_mae")
    ax_aux.set_xlabel("Эпоха")
    ax_aux.grid(alpha=0.3)
    if plotted:
        ax_aux.legend(loc="upper left", fontsize=8)

    # 5) GPU memory + util
    plotted = False
    if _has(df, "train_gpu_mem_alloc_gb"):
        ax_gpu.plot(df["epoch"], df["train_gpu_mem_alloc_gb"], label="alloc, ГБ", color="C0")
        plotted = True
    if _has(df, "train_gpu_mem_peak_gb"):
        ax_gpu.plot(df["epoch"], df["train_gpu_mem_peak_gb"], label="peak, ГБ", color="C1")
        plotted = True
    if _has(df, "train_gpu_util_pct"):
        ax_g2 = ax_gpu.twinx()
        ax_g2.plot(df["epoch"], df["train_gpu_util_pct"], color="C3",
                   ls="--", label="util, %")
        ax_g2.set_ylabel("Использование GPU, %", color="C3")
        ax_g2.set_ylim(0, 100)
        ax_g2.tick_params(axis="y", labelcolor="C3")
        ax_g2.legend(loc="upper right", fontsize=8)
        plotted = True
    if not plotted:
        ax_gpu.text(0.5, 0.5, "Нет GPU-метрик", transform=ax_gpu.transAxes,
                    ha="center", va="center", color="0.5")
    ax_gpu.set_title("GPU: память и загрузка")
    ax_gpu.set_xlabel("Эпоха")
    ax_gpu.set_ylabel("Память, ГБ")
    ax_gpu.grid(alpha=0.3)
    if plotted:
        ax_gpu.legend(loc="upper left", fontsize=8)

    # 6) Wall time per step + dataload fraction
    plotted = False
    for col, lab, color in [
        ("train_step_time_s", "step time, c", "C0"),
        ("train_compute_time_s", "compute, c", "C1"),
        ("train_data_time_s", "data, c", "C2"),
    ]:
        if col in df.columns:
            _plot(ax_time, df, col, lab, color=color)
            plotted = True
    if "train_dataload_fraction" in df.columns:
        ax_t2 = ax_time.twinx()
        ax_t2.plot(df["epoch"], df["train_dataload_fraction"] * 100,
                   color="C3", ls="--", label="data load, %")
        ax_t2.set_ylim(0, 100)
        ax_t2.set_ylabel("dataload fraction, %", color="C3")
        ax_t2.tick_params(axis="y", labelcolor="C3")
        ax_t2.legend(loc="upper right", fontsize=8)
        plotted = True
    if not plotted:
        ax_time.text(0.5, 0.5, "Нет тайминговых метрик", transform=ax_time.transAxes,
                     ha="center", va="center", color="0.5")
    ax_time.set_title("Тайминг шага (train)")
    ax_time.set_xlabel("Эпоха")
    ax_time.set_ylabel("Секунды")
    ax_time.grid(alpha=0.3)
    if plotted:
        ax_time.legend(loc="upper left", fontsize=8)

    fig.suptitle(title or os.path.basename(os.path.dirname(os.path.abspath(csv_path))),
                 fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

DEFAULT_COMPARE_METRICS: Sequence[Tuple[str, str]] = (
    ("train_loss", "Train loss"),
    ("val_miou_fg", "val mIoU (foreground)"),
    ("val_dice_fg", "val Dice (foreground)"),
    ("val_miou_fg_ang", "val mIoU (foreground, по углу)"),
    ("val_ang_mae_deg", "val MAE угла, °"),
    ("val_precision_fg", "val precision (fg)"),
    ("val_recall_fg", "val recall (fg)"),
    ("train_step_time_s", "Train step time, c"),
    ("train_gpu_mem_peak_gb", "GPU peak memory, ГБ"),
)


def figure_compare_runs(
    runs: Dict[str, str],
    metrics: Sequence[Tuple[str, str]] = DEFAULT_COMPARE_METRICS,
    n_cols: int = 3,
):
    """Compare multiple ``metrics.csv`` files on the same axes per metric.

    Parameters
    ----------
    runs
        Mapping ``{run_label: path_to_metrics_csv}``.
    metrics
        Iterable of ``(column, title)`` pairs to draw. Columns missing in
        a run are silently skipped for that run.
    """
    dfs = {label: load_metrics(p) for label, p in runs.items()}
    n = len(metrics)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // n_cols, idx % n_cols]
        any_plotted = False
        for label, df in dfs.items():
            if col in df.columns and df[col].notna().any():
                ax.plot(df["epoch"], df[col], label=label, lw=1.7)
                any_plotted = True
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.3)
        if not any_plotted:
            ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes,
                    color="0.6")
        else:
            ax.legend(fontsize=8)

    # Hide unused
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_axis_off()

    fig.suptitle("Сравнение моделей по обучающим логам", fontsize=14)
    fig.tight_layout()
    return fig


def best_epoch_summary(csv_path: str, target: str = "val_miou_fg") -> Dict[str, float]:
    """Return ``{best_epoch, best_value, last_value}`` for a metric column."""
    df = load_metrics(csv_path)
    if target not in df.columns:
        raise KeyError(f"{target!r} not in {csv_path}")
    s = df[target]
    best_idx = int(s.idxmax())
    return {
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_value": float(s.max()),
        "last_epoch": int(df["epoch"].iloc[-1]),
        "last_value": float(s.iloc[-1]),
    }
