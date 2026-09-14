"""Visualization module for EXP-06 Temporal Drift, Shifting Prevalences, and Regime Shifts."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.common.logger import get_logger

logger = get_logger("Module5.DriftVisualization")


def _add_shock_period_shading(ax: plt.Axes, min_ts: int = 40, max_ts: int = 49) -> None:
    """Adds standard shaded region for darknet shock period (t=43..46)."""
    ax.axvspan(42.5, 46.5, color="#ffcccc", alpha=0.45, label="Darknet Market Shock Window (t=43..46)")


def plot_f1_by_timestep(
    metrics: List[Dict],
    xgb_metrics: Optional[List[Dict]],
    save_path: Union[str, Path],
    title: str = "EXP-06: Illicit F1-Score Across Temporal Test Timesteps (t=40..49)"
) -> None:
    """Plots F1-score across individual test timesteps."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ts = [m["timestep"] for m in metrics]
    f1_sage = [m["f1_illicit"] for m in metrics]

    ax = plt.gca()
    _add_shock_period_shading(ax)

    plt.plot(ts, f1_sage, marker="o", lw=2.5, color="#e6ab02", label="GraphSAGE (frozen tau*=0.5517)")

    if xgb_metrics:
        f1_xgb = [m["f1_illicit"] for m in xgb_metrics]
        plt.plot(ts, f1_xgb, marker="s", lw=2.2, linestyle="--", color="#d95f02", label="XGBoost Baseline (frozen tau*=0.9014)")

    # Data labels
    for x_val, y_val in zip(ts, f1_sage):
        plt.annotate(
            f"{y_val:.2f}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.xlabel("Timestep (2-week discrete Bitcoin temporal intervals)", fontsize=11, fontweight="bold")
    plt.ylabel("Illicit Class F1-Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(ts, [f"t={t}" for t in ts], fontsize=10, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved F1 by timestep plot to {p}")


def plot_pr_auc_by_timestep(
    metrics: List[Dict],
    xgb_metrics: Optional[List[Dict]],
    save_path: Union[str, Path],
    title: str = "EXP-06: PR-AUC Across Temporal Test Timesteps (t=40..49)"
) -> None:
    """Plots PR-AUC across individual test timesteps."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ts = [m["timestep"] for m in metrics]
    prauc_sage = [m["pr_auc"] for m in metrics]

    ax = plt.gca()
    _add_shock_period_shading(ax)

    plt.plot(ts, prauc_sage, marker="o", lw=2.5, color="#7570b3", label="GraphSAGE PR-AUC")

    if xgb_metrics:
        prauc_xgb = [m["pr_auc"] for m in xgb_metrics]
        plt.plot(ts, prauc_xgb, marker="s", lw=2.2, linestyle="--", color="#d95f02", label="XGBoost Baseline PR-AUC")

    for x_val, y_val in zip(ts, prauc_sage):
        plt.annotate(
            f"{y_val:.2f}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.xlabel("Timestep (2-week discrete Bitcoin temporal intervals)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision-Recall AUC (PR-AUC)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(ts, [f"t={t}" for t in ts], fontsize=10, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved PR-AUC by timestep plot to {p}")


def plot_precision_recall_by_timestep(
    metrics: List[Dict],
    save_path: Union[str, Path],
    title: str = "EXP-06: Precision vs. Recall Across Temporal Test Timesteps (GraphSAGE)"
) -> None:
    """Plots precision and recall curves across timesteps."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ts = [m["timestep"] for m in metrics]
    prec = [m["precision_illicit"] for m in metrics]
    rec = [m["recall_illicit"] for m in metrics]

    ax = plt.gca()
    _add_shock_period_shading(ax)

    plt.plot(ts, prec, marker="^", lw=2.3, color="#2b5c8f", label="Precision (Illicit)")
    plt.plot(ts, rec, marker="v", lw=2.3, color="#1b9e77", label="Recall (Illicit)")

    plt.xlabel("Timestep", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(ts, [f"t={t}" for t in ts], fontsize=10, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved Precision vs Recall plot to {p}")


def plot_illicit_prevalence(
    metrics: List[Dict],
    save_path: Union[str, Path],
    title: str = "EXP-06: Illicit Fraud Counts & Percentage Prevalence Across Test Timesteps"
) -> None:
    """Plots dual-axis chart showing raw illicit counts and percentage prevalence."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ts = [m["timestep"] for m in metrics]
    counts = [m["illicit_count"] for m in metrics]
    prevs = [m["illicit_prevalence"] * 100 for m in metrics]

    _add_shock_period_shading(ax1)

    color1 = "#2b5c8f"
    bars = ax1.bar(np.array(ts) - 0.15, counts, width=0.3, color=color1, alpha=0.75, label="Raw Illicit Count (left axis)")
    ax1.set_xlabel("Timestep", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Illicit Transaction Count", color=color1, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(ts)
    ax1.set_xticklabels([f"t={t}" for t in ts], fontweight="bold")

    ax2 = ax1.twinx()
    color2 = "#d95f02"
    ax2.plot(ts, prevs, color=color2, marker="D", lw=2.5, label="Illicit Prevalence % (right axis)")
    ax2.set_ylabel("Illicit Prevalence (%)", color=color2, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    for x_val, y_val in zip(ts, prevs):
        ax2.annotate(
            f"{y_val:.1f}%",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="#b34700",
            fontweight="bold",
        )

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved illicit prevalence plot to {p}")


def plot_drift_summary_dashboard(
    drift_data: Dict,
    save_path: Union[str, Path],
    title: str = "EXP-06: Temporal Drift & Distribution Shift Comprehensive Dashboard"
) -> None:
    """Generates a 4-panel dashboard summarizing temporal drift, confusion trends, and regime comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    metrics = drift_data["per_timestep_metrics"]
    ts = [m["timestep"] for m in metrics]

    # Panel 1: F1 and PR-AUC trends
    ax1 = axes[0, 0]
    _add_shock_period_shading(ax1)
    ax1.plot(ts, [m["f1_illicit"] for m in metrics], marker="o", color="#d95f02", lw=2.2, label="Illicit F1")
    ax1.plot(ts, [m["pr_auc"] for m in metrics], marker="s", color="#7570b3", lw=2.2, label="PR-AUC")
    ax1.set_title("Detection Performance Across Test Timesteps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_xticks(ts)
    ax1.set_xticklabels([f"t={t}" for t in ts])
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower left", fontsize=9)

    # Panel 2: False Positives vs False Negatives
    ax2 = axes[0, 1]
    _add_shock_period_shading(ax2)
    fps = [m["false_positives"] for m in metrics]
    fns = [m["false_negatives"] for m in metrics]
    ax2.bar(np.array(ts) - 0.2, fps, width=0.4, color="#386cb0", label="False Positives (FP)", alpha=0.85)
    ax2.bar(np.array(ts) + 0.2, fns, width=0.4, color="#e41a1c", label="False Negatives (FN)", alpha=0.85)
    ax2.set_title("Classification Error Distribution (FP vs. FN)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontweight="bold")
    ax2.set_xticks(ts)
    ax2.set_xticklabels([f"t={t}" for t in ts])
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Period Regime Comparison (Pre, Shock, Post)
    ax3 = axes[1, 0]
    p_data = drift_data["period_analysis"]
    periods = ["pre_shock", "shock_period", "post_shock"]
    period_names = ["Pre-Shock\n(t=40..42)", "Shock Window\n(t=43..46)", "Post-Shock\n(t=47..49)"]
    f1_means = [p_data[p]["mean_f1"] for p in periods]
    prauc_means = [p_data[p]["mean_pr_auc"] for p in periods]
    prev_means = [p_data[p]["mean_illicit_prevalence"] * 100 for p in periods]

    x_p = np.arange(len(periods))
    width = 0.25
    r1 = ax3.bar(x_p - width, f1_means, width, color="#d95f02", label="Mean Illicit F1")
    r2 = ax3.bar(x_p, prauc_means, width, color="#7570b3", label="Mean PR-AUC")
    r3 = ax3.bar(x_p + width, [p / 10.0 for p in prev_means], width, color="#1b9e77", label="Mean Prevalence (/10%)")
    ax3.set_title("Regime Averages (Pre-Shock vs. Shock vs. Post-Shock)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_p)
    ax3.set_xticklabels(period_names, fontweight="bold")
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc="upper right", fontsize=9)

    # Panel 4: Precision vs Recall by Timestep
    ax4 = axes[1, 1]
    _add_shock_period_shading(ax4)
    ax4.plot(ts, [m["precision_illicit"] for m in metrics], marker="^", color="#2b5c8f", lw=2.2, label="Precision")
    ax4.plot(ts, [m["recall_illicit"] for m in metrics], marker="v", color="#1b9e77", lw=2.2, label="Recall")
    ax4.set_title("Precision & Recall Trajectories", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Score", fontweight="bold")
    ax4.set_xticks(ts)
    ax4.set_xticklabels([f"t={t}" for t in ts])
    ax4.set_ylim([0, 1.05])
    ax4.legend(loc="lower left", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved drift summary dashboard to {p}")
