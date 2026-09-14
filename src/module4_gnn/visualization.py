"""Visualization tools for fraud detection model curves and comparisons."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.common.logger import get_logger

logger = get_logger("Module4.Visualization")


def plot_pr_curves(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Union[str, Path],
    title: str = "Precision-Recall Curves (Test Set: t=40..49)"
) -> None:
    """Plots and saves Precision-Recall curves for evaluated models."""
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "random_forest": "#2b5c8f",
        "xgboost": "#d95f02",
        "lightgbm": "#7570b3",
        "mlp": "#1b9e77",
        "gcn": "#e7298a",
        "graphsage": "#e6ab02",
        "gat": "#a6761d",
    }
    display_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "gcn": "GCN",
        "graphsage": "GraphSAGE",
        "gat": "Multi-Head GAT",
    }

    pos_ratio = np.sum(y_test == 1) / len(y_test)
    plt.axhline(
        y=pos_ratio,
        color="gray",
        linestyle="--",
        label=f"No-Skill Baseline (PR-AUC={pos_ratio:.4f})"
    )

    for model_name, y_probs in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, y_probs, pos_label=1)
        pr_auc = auc(rec, prec)
        color = colors.get(model_name, None)
        label_text = f"{display_names.get(model_name, model_name.upper())} (PR-AUC = {pr_auc:.4f})"
        plt.plot(rec, prec, lw=2.2, color=color, label=label_text)

    plt.xlabel("Recall (Illicit Fraud Class)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Illicit Fraud Class)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved PR curves plot to {p}")


def plot_roc_curves(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Union[str, Path],
    title: str = "ROC Curves (Test Set: t=40..49)"
) -> None:
    """Plots and saves ROC curves for evaluated models."""
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "random_forest": "#2b5c8f",
        "xgboost": "#d95f02",
        "lightgbm": "#7570b3",
        "mlp": "#1b9e77",
        "gcn": "#e7298a",
        "graphsage": "#e6ab02",
        "gat": "#a6761d",
    }
    display_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "gcn": "GCN",
        "graphsage": "GraphSAGE",
        "gat": "Multi-Head GAT",
    }

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (ROC-AUC = 0.5000)")

    for model_name, y_probs in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        color = colors.get(model_name, None)
        label_text = f"{display_names.get(model_name, model_name.upper())} (ROC-AUC = {roc_auc:.4f})"
        plt.plot(fpr, tpr, lw=2.2, color=color, label=label_text)

    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved ROC curves plot to {p}")


def plot_gnn_training_curves(
    training_histories: Dict[str, Dict[str, List[float]]],
    save_path: Union[str, Path],
    title: str = "EXP-02: GNN Training Convergence (Loss & Validation PR-AUC)"
) -> None:
    """Plots training loss and validation PR-AUC curves across epochs for GNN models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {"gcn": "#e7298a", "graphsage": "#e6ab02", "gat": "#a6761d"}

    for name, hist in training_histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = colors.get(name, None)
        ax1.plot(epochs, hist["train_loss"], lw=2.0, color=c, label=f"{name.upper()} Train Loss")
        ax2.plot(epochs, hist["val_pr_auc"], lw=2.0, color=c, label=f"{name.upper()} Val PR-AUC")

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Focal Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Training Loss Across Timesteps", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation PR-AUC", fontsize=11, fontweight="bold")
    ax2.set_title("Validation PR-AUC Across Epochs", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved GNN training curves to {p}")


def plot_feature_ablation_comparison(
    exp03_results: Dict,
    save_path: Union[str, Path],
    title: str = "EXP-03: Feature Configuration Ablation (GraphSAGE)"
) -> None:
    """Plots bar chart comparing feature configurations in EXP-03."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    configs = list(exp03_results["configurations"].keys())
    display_names = [f"{c}\n({exp03_results['configurations'][c]['feature_count']} feats)" for c in configs]

    metrics = ["f1_illicit", "pr_auc", "precision_illicit", "recall_illicit"]
    labels = ["Test Illicit F1", "Test PR-AUC", "Test Precision", "Test Recall"]
    colors = ["#d95f02", "#7570b3", "#2b5c8f", "#1b9e77"]

    x = np.arange(len(configs))
    width = 0.18

    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [exp03_results["configurations"][c]["test"][m] for c in configs]
        rects = plt.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=col)
        for rect in rects:
            h = rect.get_height()
            plt.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, display_names, fontsize=10, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved feature ablation plot to {p}")


def plot_loss_comparison(
    exp04_results: Dict,
    save_path: Union[str, Path],
    title: str = "EXP-04: Class Imbalance Loss Comparison (GraphSAGE)"
) -> None:
    """Plots bar chart comparing loss functions in EXP-04."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    losses = list(exp04_results["loss_functions"].keys())
    display_names = [l.replace("_", " ").title() for l in losses]

    metrics = ["f1_illicit", "pr_auc", "precision_illicit", "recall_illicit"]
    labels = ["Test Illicit F1", "Test PR-AUC", "Test Precision", "Test Recall"]
    colors = ["#d95f02", "#7570b3", "#2b5c8f", "#1b9e77"]

    x = np.arange(len(losses))
    width = 0.18

    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [exp04_results["loss_functions"][l]["test"][m] for l in losses]
        rects = plt.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=col)
        for rect in rects:
            h = rect.get_height()
            plt.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, display_names, fontsize=10, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved loss comparison plot to {p}")


def plot_full_model_comparison(
    models_data: List[Dict],
    save_path: Union[str, Path],
    title: str = "Full Benchmark: Tabular Baselines vs. Graph Neural Networks (Test Set: t=40..49)"
) -> None:
    """Plots a comprehensive bar chart comparing all tabular baselines and GNN architectures."""
    plt.figure(figsize=(13, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    names = [m["name"] for m in models_data]
    f1_scores = [m["f1_illicit"] for m in models_data]
    pr_aucs = [m["pr_auc"] for m in models_data]
    precisions = [m["precision"] for m in models_data]
    recalls = [m["recall"] for m in models_data]

    x = np.arange(len(names))
    width = 0.18

    r1 = plt.bar(x - 1.5 * width, f1_scores, width, label="Illicit F1", color="#d95f02")
    r2 = plt.bar(x - 0.5 * width, pr_aucs, width, label="PR-AUC", color="#7570b3")
    r3 = plt.bar(x + 0.5 * width, precisions, width, label="Precision", color="#2b5c8f")
    r4 = plt.bar(x + 1.5 * width, recalls, width, label="Recall", color="#1b9e77")

    # Annotate F1 scores
    for rect in r1:
        height = rect.get_height()
        plt.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(x, names, fontsize=11, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved full model comparison plot to {p}")
