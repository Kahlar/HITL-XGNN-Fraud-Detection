"""Visualization module for EXP-07 Human-in-the-Loop Active Learning benchmarks."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.common.logger import get_logger

logger = get_logger("Module7.Visualization")


def plot_f1_vs_budget(
    benchmark_data: Dict[str, Dict[float, float]],
    save_path: Union[str, Path],
    title: str = "EXP-07: Test Illicit F1-Score vs. Human Feedback Budget"
) -> None:
    """Plots Test F1 curve across feedback budgets (0%, 5%, 10%, 20%) by strategy."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "no_feedback": "#7f7f7f",
        "random": "#2b5c8f",
        "uncertainty": "#e6ab02",
        "entropy": "#7570b3",
        "high_risk": "#d95f02",
        "combined_active": "#1b9e77",
    }
    markers = {"no_feedback": "x", "random": "o", "uncertainty": "^", "entropy": "s", "high_risk": "D", "combined_active": "P"}

    for strat, b_dict in benchmark_data.items():
        budgets = sorted(b_dict.keys())
        f1_scores = [b_dict[b] for b in budgets]
        x_pcts = [b * 100 for b in budgets]
        c = colors.get(strat, "#333333")
        m = markers.get(strat, "o")
        label_text = strat.replace("_", " ").title()
        plt.plot(x_pcts, f1_scores, marker=m, lw=2.2, color=c, label=label_text)

    plt.xlabel("Feedback Budget (% of Validation Candidates Reviewed)", fontsize=11, fontweight="bold")
    plt.ylabel("Test Illicit F1-Score (t=40..49)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks([0, 5, 10, 20], ["0% (No Feedback)", "5% (~274 tx)", "10% (~548 tx)", "20% (~1097 tx)"], fontweight="bold")
    plt.ylim([0.45, 0.70])
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved F1 vs budget plot to {p}")


def plot_pr_auc_vs_budget(
    benchmark_data: Dict[str, Dict[float, float]],
    save_path: Union[str, Path],
    title: str = "EXP-07: Test PR-AUC vs. Human Feedback Budget"
) -> None:
    """Plots Test PR-AUC curve across feedback budgets."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "no_feedback": "#7f7f7f",
        "random": "#2b5c8f",
        "uncertainty": "#e6ab02",
        "entropy": "#7570b3",
        "high_risk": "#d95f02",
        "combined_active": "#1b9e77",
    }
    markers = {"no_feedback": "x", "random": "o", "uncertainty": "^", "entropy": "s", "high_risk": "D", "combined_active": "P"}

    for strat, b_dict in benchmark_data.items():
        budgets = sorted(b_dict.keys())
        pr_scores = [b_dict[b] for b in budgets]
        x_pcts = [b * 100 for b in budgets]
        c = colors.get(strat, "#333333")
        m = markers.get(strat, "o")
        plt.plot(x_pcts, pr_scores, marker=m, lw=2.2, color=c, label=strat.replace("_", " ").title())

    plt.xlabel("Feedback Budget (% of Validation Candidates Reviewed)", fontsize=11, fontweight="bold")
    plt.ylabel("Test PR-AUC (t=40..49)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks([0, 5, 10, 20], ["0% (No Feedback)", "5% (~274 tx)", "10% (~548 tx)", "20% (~1097 tx)"], fontweight="bold")
    plt.ylim([0.40, 0.60])
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved PR-AUC vs budget plot to {p}")


def plot_sample_efficiency(
    efficiencies: List[Dict],
    save_path: Union[str, Path],
    budget_target: float = 0.10,
    title: str = "EXP-07: Sample Efficiency (F1 Gain per 1,000 Reviewed Transactions at 10% Budget)"
) -> None:
    """Plots bar chart of sample efficiency metrics."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    filtered = [e for e in efficiencies if e["budget_ratio"] == budget_target and e["strategy"] != "no_feedback"]
    strats = [e["strategy"].replace("_", " ").title() for e in filtered]
    f1_effs = [e["f1_efficiency_per_1k"] for e in filtered]
    prauc_effs = [e["pr_auc_efficiency_per_1k"] for e in filtered]

    x = np.arange(len(strats))
    width = 0.35

    b1 = plt.bar(x - width/2, f1_effs, width, label="F1 Delta / 1k Samples", color="#d95f02", alpha=0.85)
    b2 = plt.bar(x + width/2, prauc_effs, width, label="PR-AUC Delta / 1k Samples", color="#7570b3", alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        plt.annotate(
            f"{h:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.ylabel("Metric Improvement per 1,000 Reviews", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, strats, fontweight="bold")
    plt.axhline(0, color="gray", linestyle="--", lw=1)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved sample efficiency plot to {p}")


def plot_uncertainty_distribution(
    query_stats: List[Dict],
    save_path: Union[str, Path],
    title: str = "EXP-07: Query Entropy Distribution Across Active Learning Strategies"
) -> None:
    """Plots mean prediction entropy across query strategies."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    strats = [s["strategy"].replace("_", " ").title() for s in query_stats]
    means = [s["mean_entropy_bits"] for s in query_stats]
    stds = [s["std_entropy_bits"] for s in query_stats]

    colors = ["#2b5c8f", "#e6ab02", "#7570b3", "#d95f02", "#1b9e77"][:len(strats)]
    bars = plt.bar(strats, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.45)

    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.3f} bits",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylabel("Mean Shannon Entropy (bits)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved uncertainty distribution plot to {p}")


def plot_temporal_comparison(
    ts_base_f1: List[float],
    ts_active_f1: List[float],
    timesteps: List[int],
    save_path: Union[str, Path],
    metric_name: str = "F1-Score",
    title: str = "EXP-07: Out-of-Time Robustness — Base Model vs. Retrained Active Learner"
) -> None:
    """Plots per-timestep performance comparison highlighting the darknet shock window."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ax = plt.gca()
    ax.axvspan(42.5, 46.5, color="#ffcccc", alpha=0.45, label="Darknet Market Shock Window (t=43..46)")

    plt.plot(timesteps, ts_base_f1, marker="o", lw=2.2, color="#7f7f7f", linestyle="--", label="Base GraphSAGE (No Feedback)")
    plt.plot(timesteps, ts_active_f1, marker="D", lw=2.5, color="#1b9e77", label="Combined Active Retrained (20% Budget)")

    for x_val, y_val in zip(timesteps, ts_active_f1):
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
    plt.ylabel(f"Illicit Class {metric_name}", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(timesteps, [f"t={t}" for t in timesteps], fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved temporal comparison plot to {p}")


def plot_feedback_class_distribution(
    query_stats: List[Dict],
    save_path: Union[str, Path],
    title: str = "EXP-07: Query Composition — Illicit vs. Licit Feedback Proportions (10% Budget)"
) -> None:
    """Plots proportion of illicit vs licit samples queried by each strategy."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    strats = [s["strategy"].replace("_", " ").title() for s in query_stats]
    illicit_pcts = [s["illicit_percentage"] for s in query_stats]
    licit_pcts = [100.0 - s["illicit_percentage"] for s in query_stats]

    x = np.arange(len(strats))
    width = 0.45

    plt.bar(x, illicit_pcts, width, label="Illicit Fraud Query %", color="#d95f02", alpha=0.85)
    plt.bar(x, licit_pcts, width, bottom=illicit_pcts, label="Licit Clean Query %", color="#2b5c8f", alpha=0.85)

    for i, p in enumerate(illicit_pcts):
        plt.annotate(
            f"{p:.1f}% Illicit",
            xy=(x[i], p / 2 if p > 10 else 10),
            ha="center",
            va="center",
            color="white" if p > 10 else "black",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylabel("Query Composition (%)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, strats, fontweight="bold")
    plt.ylim([0, 105])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved feedback class distribution plot to {p}")
