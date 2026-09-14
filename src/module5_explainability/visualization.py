"""Visualization tools for explanation fidelity, feature attribution, sparsity, latency, and subgraph graphs."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.common.logger import get_logger
from src.module5_explainability.gnn_explainer import TransactionExplanation

logger = get_logger("Module5.XAIVisualization")


def plot_fidelity_plus_comparison(
    method_fidelities: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "EXP-05: Explanation Faithfulness — Fidelity+ (Sufficiency)"
) -> None:
    """Plots Fidelity+ comparison across explanation methods (GNNExplainer vs GAT vs Random)."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    methods = list(method_fidelities.keys())
    display_names = [m.replace("_", " ").title() for m in methods]
    colors = ["#d95f02", "#7570b3", "#2b5c8f"][:len(methods)]

    means = [float(np.mean(method_fidelities[m])) for m in methods]
    stds = [float(np.std(method_fidelities[m])) for m in methods]

    bars = plt.bar(display_names, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylabel("Fidelity+ (p_orig - p_removed)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved Fidelity+ plot to {p}")


def plot_fidelity_minus_comparison(
    method_fidelities: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "EXP-05: Explanation Faithfulness — Fidelity- (Necessity)"
) -> None:
    """Plots Fidelity- comparison across explanation methods."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    methods = list(method_fidelities.keys())
    display_names = [m.replace("_", " ").title() for m in methods]
    colors = ["#d95f02", "#7570b3", "#2b5c8f"][:len(methods)]

    means = [float(np.mean(method_fidelities[m])) for m in methods]
    stds = [float(np.std(method_fidelities[m])) for m in methods]

    bars = plt.bar(display_names, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylabel("Fidelity- (p_orig - p_expl)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved Fidelity- plot to {p}")


def plot_sparsity_tradeoff(
    budget_data: Dict[str, Dict[float, Dict[str, float]]],
    save_path: Union[str, Path],
    title: str = "EXP-05: Explanation Fidelity vs. Edge Sparsity Across Budget Constraints"
) -> None:
    """Plots fidelity vs budget/sparsity trade-off curves."""
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {"gnn_explainer": "#d95f02", "gat_attention": "#7570b3", "random": "#2b5c8f"}

    for method, b_dict in budget_data.items():
        budgets = sorted(b_dict.keys())
        fid_plus = [b_dict[b]["mean_fid_plus"] for b in budgets]
        sparsities = [b_dict[b]["mean_edge_sparsity"] * 100 for b in budgets]
        c = colors.get(method, "#333333")
        plt.plot(sparsities, fid_plus, marker="o", lw=2.4, color=c, label=f"{method.replace('_', ' ').title()}")

    plt.xlabel("Edge Sparsity (% of non-explanation edges removed)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Fidelity+ (Sufficiency Drop)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved sparsity trade-off plot to {p}")


def plot_explanation_stability(
    jaccard_scores: List[float],
    save_path: Union[str, Path],
    title: str = "EXP-05: Explanation Stability — Top-k Edge Jaccard Similarity Across Random Seeds"
) -> None:
    """Plots distribution of explanation stability (Jaccard similarity)."""
    plt.figure(figsize=(8.5, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.hist(jaccard_scores, bins=10, range=(0, 1.0), color="#1b9e77", edgecolor="black", alpha=0.85)
    mean_j = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    plt.axvline(mean_j, color="#d95f02", linestyle="--", lw=2.5, label=f"Mean Jaccard = {mean_j:.4f}")

    plt.xlabel("Pairwise Top-k Edge Jaccard Similarity", fontsize=11, fontweight="bold")
    plt.ylabel("Target Transaction Count", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved stability plot to {p}")


def plot_feature_attribution_distribution(
    explanations: List[TransactionExplanation],
    save_path: Union[str, Path],
    title: str = "EXP-05: Feature Attribution Breakdown (Local vs. Aggregated Dimensions)"
) -> None:
    """Plots the relative importance of local features (0..92) vs aggregated features (93..164)."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    cats = ["TP", "FN", "FP", "TN"]
    cat_expls = {c: [e for e in explanations if e.category == c] for c in cats}

    local_means = []
    agg_means = []
    active_cats = []

    for c in cats:
        items = cat_expls[c]
        if items:
            active_cats.append(c)
            local_means.append(float(np.mean([e.feature_summary["local_importance_ratio"] for e in items])))
            agg_means.append(float(np.mean([e.feature_summary["aggregated_importance_ratio"] for e in items])))

    x = np.arange(len(active_cats))
    width = 0.35

    plt.bar(x - width/2, local_means, width, label="Local Features (0..92)", color="#2b5c8f", alpha=0.85)
    plt.bar(x + width/2, agg_means, width, label="Aggregated Features (93..164)", color="#e6ab02", alpha=0.85)

    plt.ylabel("Mean Normalized Attribution Ratio", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, [f"Category: {c}" for c in active_cats], fontweight="bold")
    plt.ylim([0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved feature attribution plot to {p}")


def plot_edge_attribution_distribution(
    explanations: List[TransactionExplanation],
    save_path: Union[str, Path],
    title: str = "EXP-05: Edge Attribution Weight Distribution Across 2-Hop Subgraphs"
) -> None:
    """Plots histogram of normalized edge attribution weights."""
    plt.figure(figsize=(8.5, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    all_edge_weights = []
    for e in explanations:
        all_edge_weights.extend(e.edge_mask)

    plt.hist(all_edge_weights, bins=25, color="#7570b3", edgecolor="black", alpha=0.85)
    plt.xlabel("Raw Edge Attribution Weight (GNNExplainer Sigmoid Mask)", fontsize=11, fontweight="bold")
    plt.ylabel("Edge Count", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved edge attribution plot to {p}")


def plot_latency_distribution(
    latencies_ms: List[float],
    save_path: Union[str, Path],
    title: str = "EXP-05: GNNExplainer Inference Latency Distribution (ms per transaction)"
) -> None:
    """Plots explanation generation runtime latency distribution."""
    plt.figure(figsize=(8.5, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.hist(latencies_ms, bins=15, color="#386cb0", edgecolor="black", alpha=0.85)
    mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    plt.axvline(mean_lat, color="#d95f02", linestyle="--", lw=2.2, label=f"Mean Latency = {mean_lat:.1f} ms")
    plt.axvline(p95_lat, color="#e41a1c", linestyle=":", lw=2.2, label=f"P95 Latency = {p95_lat:.1f} ms")

    plt.xlabel("Explanation Generation Latency (ms)", fontsize=11, fontweight="bold")
    plt.ylabel("Sample Count", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved latency distribution plot to {p}")


def plot_temporal_fidelity(
    regime_fidelities: Dict[str, Dict[str, float]],
    save_path: Union[str, Path],
    title: str = "EXP-05: Explanation Fidelity Across Temporal Distribution Shift Regimes"
) -> None:
    """Plots Fidelity+ and Fidelity- across Pre-shock, Shock, and Post-shock periods."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    regimes = ["pre_shock", "shock_period", "post_shock"]
    display_names = ["Pre-Shock\n(t=40..42)", "Shock Window\n(t=43..46)", "Post-Shock\n(t=47..49)"]

    fid_plus = [regime_fidelities[r]["mean_fid_plus"] for r in regimes]
    fid_minus = [regime_fidelities[r]["mean_fid_minus"] for r in regimes]

    x = np.arange(len(regimes))
    width = 0.35

    plt.bar(x - width/2, fid_plus, width, label="Fidelity+ (Sufficiency)", color="#d95f02", alpha=0.85)
    plt.bar(x + width/2, fid_minus, width, label="Fidelity- (Necessity)", color="#7570b3", alpha=0.85)

    plt.ylabel("Mean Fidelity Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, display_names, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p)
    plt.close()
    logger.info(f"Saved temporal fidelity plot to {p}")


def plot_representative_subgraph_explanation(
    explanation: TransactionExplanation,
    save_path: Union[str, Path],
    top_k_edges: int = 5,
) -> None:
    """
    Renders a directed graph visualizer of the 2-hop computational subgraph,
    highlighting the target transaction node and top explanatory edges.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    plt.style.use("default")

    G = nx.DiGraph()

    # Collect nodes and edges
    target_loc = explanation.local_node_index
    top_edge_indices = {e["edge_index_in_subgraph"] for e in explanation.top_edges[:top_k_edges]}

    # Add edges
    for e in explanation.top_edges:
        src = e["source_local_idx"]
        dst = e["target_local_idx"]
        is_top = e["edge_index_in_subgraph"] in top_edge_indices
        G.add_edge(src, dst, weight=e["normalized_importance"], is_top=is_top)

    if G.number_of_nodes() == 0:
        G.add_node(target_loc)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.8)

    # Draw background edges
    bg_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_top", False)]
    top_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_top", False)]

    if bg_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=bg_edges, edge_color="#cccccc",
            arrows=True, arrowstyle="-|>", arrowsize=12, width=1.0, alpha=0.6,
        )

    # Draw top explanation edges
    if top_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=top_edges, edge_color="#e41a1c",
            arrows=True, arrowstyle="-|>", arrowsize=18, width=2.8, alpha=0.95,
        )

    # Draw regular nodes
    reg_nodes = [n for n in G.nodes() if n != target_loc]
    if reg_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=reg_nodes, node_color="#386cb0",
            node_size=350, alpha=0.8,
        )

    # Draw target node prominently
    target_color = "#e41a1c" if explanation.ground_truth == 1 else "#4daf4a"
    nx.draw_networkx_nodes(
        G, pos, nodelist=[target_loc], node_color=target_color,
        node_size=800, edgecolors="black", linewidths=2.5,
    )

    # Node labels
    labels = {target_loc: f"TARGET\n{explanation.tx_id[:6]}"}
    for n in reg_nodes[:10]:
        labels[n] = str(n)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

    # Title & Metadata annotation
    pred_str = "ILLICIT" if explanation.predicted_class == 1 else "LICIT"
    gt_str = "ILLICIT" if explanation.ground_truth == 1 else ("LICIT" if explanation.ground_truth == 0 else "UNKNOWN")
    meta_text = (
        f"Tx ID: {explanation.tx_id} | Timestep: t={explanation.timestep} ({explanation.period})\n"
        f"Predicted: {pred_str} (P(Fraud) = {explanation.prediction_probability:.4f}) | Ground Truth: {gt_str} [{explanation.category}]\n"
        f"Top Features: {', '.join([f['feature_name'] for f in explanation.top_features[:4]])}\n"
        f"Fidelity+ = {explanation.fidelity_plus if explanation.fidelity_plus is not None else 0.0:.4f} | Latency = {explanation.generation_latency_ms:.1f}ms"
    )

    plt.title(f"Local Computational Subgraph Explanation [{explanation.category}]", fontsize=13, fontweight="bold", pad=15)
    plt.figtext(0.5, 0.02, meta_text, ha="center", fontsize=9, bbox={"facecolor": "#f0f0f0", "alpha": 0.8, "pad": 6})
    plt.axis("off")
    plt.tight_layout()

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved representative subgraph explanation to {p}")
