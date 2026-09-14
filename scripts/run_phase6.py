"""Phase 6 Master Execution Script: EXP-05 Explainability & Fidelity Benchmark (GNNExplainer vs. GAT vs. Random)."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader
from src.module3_graph_builder.subgraph_extractor import SubgraphExtractor
from src.module4_gnn.models.graphsage import GraphSAGENet
from src.module5_explainability.attention_extractor import GATAttentionExtractor
from src.module5_explainability.fidelity import FidelityEvaluator
from src.module5_explainability.gnn_explainer import GNNExplainerEngine, TransactionExplanation
from src.module5_explainability.latency import LatencyProfiler
from src.module5_explainability.stability import ExplanationStabilityEvaluator
from src.module5_explainability.visualization import (
    plot_edge_attribution_distribution,
    plot_explanation_stability,
    plot_feature_attribution_distribution,
    plot_fidelity_minus_comparison,
    plot_fidelity_plus_comparison,
    plot_latency_distribution,
    plot_representative_subgraph_explanation,
    plot_sparsity_tradeoff,
    plot_temporal_fidelity,
)

logger = get_logger("Phase6.Runner")


def select_stratified_evaluation_targets(
    loader: TimestepGraphLoader,
    model: GraphSAGENet,
    threshold: float = 0.5517,
    targets_per_category: int = 12,
    random_seed: int = 42,
) -> List[Tuple[int, int, str, str]]:
    """
    Selects a balanced, reproducible set of target transactions across TP, FN, FP, TN categories
    and across Pre-Shock (40..42), Shock (43..46), and Post-Shock (47..49) regimes.

    Returns:
        List of (timestep, local_idx_in_timestep_graph, category, tx_id)
    """
    np.random.seed(random_seed)
    candidates: Dict[str, List[Tuple[int, int, str, str]]] = defaultdict(list)

    for ts in range(40, 50):
        data = loader.load_graph(ts)
        x = loader.get_feature_matrix(data, config="original_all")
        edge_index = data.edge_index
        y = data.y.numpy()

        labeled_mask = (y != -1)
        with torch.no_grad():
            logits = model(x, edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = (probs >= threshold).astype(int)

        for local_idx in np.where(labeled_mask)[0]:
            y_val = int(y[local_idx])
            pred_val = int(preds[local_idx])
            tx_id = data.node_tx_ids[local_idx] if hasattr(data, "node_tx_ids") and data.node_tx_ids else str(local_idx)

            if y_val == 1 and pred_val == 1:
                cat = "TP"
            elif y_val == 1 and pred_val == 0:
                cat = "FN"
            elif y_val == 0 and pred_val == 1:
                cat = "FP"
            elif y_val == 0 and pred_val == 0:
                cat = "TN"
            else:
                continue

            candidates[cat].append((ts, int(local_idx), cat, tx_id))

    # Sample reproducibly from each category
    selected: List[Tuple[int, int, str, str]] = []
    for cat in ["TP", "FN", "FP", "TN"]:
        pool = candidates[cat]
        if len(pool) <= targets_per_category:
            sampled = pool
        else:
            indices = np.linspace(0, len(pool) - 1, targets_per_category, dtype=int)
            sampled = [pool[i] for i in indices]
        selected.extend(sampled)
        logger.info(f"Category {cat:<3}: Available = {len(pool):<4} | Sampled = {len(sampled):<2}")

    return selected


def main() -> int:
    """Executes the complete Phase 6 EXP-05 Explainability & Fidelity Benchmark."""
    start_time = datetime.now()
    logger.info("=" * 85)
    logger.info("STARTING PHASE 6: EXPLAINABILITY ENGINE & FIDELITY BENCHMARK (EXP-05)")
    logger.info("=" * 85)

    graphs_dir = default_config.paths.processed_data_dir / "graphs"
    models_dir = Path("models/gnn")
    results_dir = default_config.paths.processed_data_dir / "experiments"
    explanations_dir = results_dir / "exp05_explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Frozen GraphSAGE Checkpoint
    sage_ckpt = models_dir / "graphsage_best.pt"
    logger.info(f"Loading frozen GraphSAGE model from {sage_ckpt}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphsage = GraphSAGENet.load_checkpoint(sage_ckpt, device=device)
    graphsage.eval()

    # 2. Load Frozen GAT Baseline Checkpoint if available
    gat_ckpt = models_dir / "gat_best.pt"
    gat_extractor = GATAttentionExtractor(gat_checkpoint=gat_ckpt, device=device)
    logger.info(f"GAT Model Available for Attention Extraction: {gat_extractor.is_available}")

    # 3. Initialize Infrastructure
    loader = TimestepGraphLoader(graphs_dir)
    extractor = SubgraphExtractor(k_default=2)
    explainer_engine = GNNExplainerEngine(model=graphsage, epochs=80, lr=0.01, device=device, random_seed=42)
    fidelity_eval = FidelityEvaluator(model=graphsage, device=device)
    stability_eval = ExplanationStabilityEvaluator(model=graphsage, device=device)
    latency_profiler = LatencyProfiler()

    # 4. Stratified Target Selection
    frozen_tau = 0.5517
    targets = select_stratified_evaluation_targets(
        loader=loader,
        model=graphsage,
        threshold=frozen_tau,
        targets_per_category=12,
        random_seed=42,
    )
    logger.info(f"\nTotal Selected Explanation Targets: {len(targets)}")

    # Containers for results
    all_explanations: List[TransactionExplanation] = []
    fidelity_results_by_method: Dict[str, List[float]] = defaultdict(list)
    fidelity_minus_by_method: Dict[str, List[float]] = defaultdict(list)
    budget_sparsity_tracking: Dict[str, Dict[float, Dict[str, List[float]]]] = {
        m: {b: {"fid_plus": [], "edge_sparsity": []} for b in [0.05, 0.10, 0.20, 0.30]}
        for m in (["gnn_explainer", "gat_attention", "random"] if gat_extractor.is_available else ["gnn_explainer", "random"])
    }

    stability_jaccard_scores: List[float] = []
    gat_agreement_rhos: List[float] = []
    gat_agreement_jaccards: List[float] = []

    # Category and Temporal groupings
    fidelities_by_category: Dict[str, List[float]] = defaultdict(list)
    fidelities_minus_by_category: Dict[str, List[float]] = defaultdict(list)
    fidelities_by_regime: Dict[str, Dict[str, List[float]]] = {
        "pre_shock": {"plus": [], "minus": []},
        "shock_period": {"plus": [], "minus": []},
        "post_shock": {"plus": [], "minus": []},
    }

    # 5. Execute Local Explanation & Fidelity Benchmarking
    logger.info("\n--- Generating Local Explanations & Perturbation Benchmarks ---")
    for idx, (ts, local_idx, category, tx_id) in enumerate(targets, start=1):
        ts_graph = loader.load_graph(ts)
        # Squeeze ts_graph features to 165 original_all
        ts_graph.x = loader.get_feature_matrix(ts_graph, config="original_all")

        # Extract 2-hop computational subgraph
        subgraph = extractor.extract_k_hop_subgraph(ts_graph, target=local_idx, k=2, flow="both")
        # Ensure subgraph.subgraph.x has 165 features
        if subgraph.subgraph.x.shape[1] > 165:
            subgraph.subgraph.x = subgraph.subgraph.x[:, :165]

        # Generate GNNExplainer explanation
        expl = explainer_engine.explain_subgraph(
            subgraph=subgraph,
            target_local_idx=subgraph.target_local_idx,
            timestep=ts,
            threshold=frozen_tau,
            top_k_features=15,
            top_k_edges=10,
        )

        latency_profiler.record(expl.generation_latency_ms, subgraph_nodes=subgraph.num_nodes)

        # Evaluate Fidelity for GNNExplainer
        budgets = [0.05, 0.10, 0.20, 0.30]
        gnn_fid = fidelity_eval.evaluate_explanation_faithfulness(
            data=subgraph.subgraph,
            target_local_idx=subgraph.target_local_idx,
            edge_mask=expl.edge_mask,
            feature_mask=expl.feature_mask,
            method_name="gnn_explainer",
            budget_ratios=budgets,
        )
        # 10% budget is our primary reference
        primary_fid = gnn_fid[1]
        expl.fidelity_plus = primary_fid.fidelity_plus
        expl.fidelity_minus = primary_fid.fidelity_minus
        expl.edge_sparsity = primary_fid.edge_sparsity
        expl.feature_sparsity = primary_fid.feature_sparsity

        all_explanations.append(expl)

        fidelity_results_by_method["gnn_explainer"].append(primary_fid.fidelity_plus)
        fidelity_minus_by_method["gnn_explainer"].append(primary_fid.fidelity_minus)
        for f_res in gnn_fid:
            budget_sparsity_tracking["gnn_explainer"][f_res.budget_ratio]["fid_plus"].append(f_res.fidelity_plus)
            budget_sparsity_tracking["gnn_explainer"][f_res.budget_ratio]["edge_sparsity"].append(f_res.edge_sparsity)

        # Evaluate Random Baseline
        rand_fid = fidelity_eval.evaluate_random_baseline(
            data=subgraph.subgraph,
            target_local_idx=subgraph.target_local_idx,
            budget_ratios=budgets,
            random_seed=42 + idx,
        )
        fidelity_results_by_method["random"].append(rand_fid[1].fidelity_plus)
        fidelity_minus_by_method["random"].append(rand_fid[1].fidelity_minus)
        for f_res in rand_fid:
            budget_sparsity_tracking["random"][f_res.budget_ratio]["fid_plus"].append(f_res.fidelity_plus)
            budget_sparsity_tracking["random"][f_res.budget_ratio]["edge_sparsity"].append(f_res.edge_sparsity)

        # Evaluate GAT Attention Baseline if available
        if gat_extractor.is_available:
            gat_edges = gat_extractor.extract_subgraph_attention(subgraph)
            # Create aligned edge mask from GAT attention
            gat_edge_dict = {(e.source_local_idx, e.target_local_idx): e.mean_attention for e in gat_edges}
            edge_idx_np = subgraph.subgraph.edge_index.cpu().numpy()
            gat_mask = [gat_edge_dict.get((int(edge_idx_np[0, i]), int(edge_idx_np[1, i])), 0.0) for i in range(subgraph.num_edges)]

            gat_fid = fidelity_eval.evaluate_explanation_faithfulness(
                data=subgraph.subgraph,
                target_local_idx=subgraph.target_local_idx,
                edge_mask=gat_mask,
                feature_mask=expl.feature_mask,
                method_name="gat_attention",
                budget_ratios=budgets,
            )
            fidelity_results_by_method["gat_attention"].append(gat_fid[1].fidelity_plus)
            fidelity_minus_by_method["gat_attention"].append(gat_fid[1].fidelity_minus)
            for f_res in gat_fid:
                budget_sparsity_tracking["gat_attention"][f_res.budget_ratio]["fid_plus"].append(f_res.fidelity_plus)
                budget_sparsity_tracking["gat_attention"][f_res.budget_ratio]["edge_sparsity"].append(f_res.edge_sparsity)

            # Agreement metrics
            agr = gat_extractor.compute_agreement(expl.edge_mask, gat_edges, subgraph.subgraph.edge_index, top_k=10)
            gat_agreement_rhos.append(agr["spearman_rho"])
            gat_agreement_jaccards.append(agr["top_k_jaccard_overlap"])

        # Category and Temporal Tracking
        fidelities_by_category[category].append(primary_fid.fidelity_plus)
        fidelities_minus_by_category[category].append(primary_fid.fidelity_minus)
        fidelities_by_regime[expl.period]["plus"].append(primary_fid.fidelity_plus)
        fidelities_by_regime[expl.period]["minus"].append(primary_fid.fidelity_minus)

        # Save individual structured JSON explanation artifact
        tx_json_path = explanations_dir / f"tx_{expl.tx_id}.json"
        with open(tx_json_path, "w", encoding="utf-8") as f:
            json.dump(expl.to_dict(), f, indent=2)

        # Run Stability on a subset (every 4th target)
        if idx % 4 == 0:
            stab = stability_eval.evaluate_node_stability(
                subgraph=subgraph,
                target_local_idx=subgraph.target_local_idx,
                timestep=ts,
                seeds=[42, 123, 999],
                top_k_edges=10,
                epochs=60,
            )
            stability_jaccard_scores.append(stab.mean_edge_jaccard)

        logger.info(
            f"[{idx:02d}/{len(targets)}] TS {ts:02d} | Tx: {tx_id[:8]}.. | Cat: {category:<2} | "
            f"P(Fraud)={expl.prediction_probability:.4f} | Fid+={primary_fid.fidelity_plus:+.4f} | "
            f"Fid-={primary_fid.fidelity_minus:+.4f} | Latency={expl.generation_latency_ms:.1f}ms"
        )

    # 6. Aggregate Summary Statistics
    latency_summary = latency_profiler.summarize()

    budget_summary_for_plot: Dict[str, Dict[float, Dict[str, float]]] = {}
    for m, b_dict in budget_sparsity_tracking.items():
        budget_summary_for_plot[m] = {}
        for b, v_dict in b_dict.items():
            budget_summary_for_plot[m][b] = {
                "mean_fid_plus": round(float(np.mean(v_dict["fid_plus"])), 4),
                "mean_edge_sparsity": round(float(np.mean(v_dict["edge_sparsity"])), 4),
            }

    regime_summary: Dict[str, Dict[str, float]] = {}
    for r in ["pre_shock", "shock_period", "post_shock"]:
        regime_summary[r] = {
            "mean_fid_plus": round(float(np.mean(fidelities_by_regime[r]["plus"])), 4) if fidelities_by_regime[r]["plus"] else 0.0,
            "mean_fid_minus": round(float(np.mean(fidelities_by_regime[r]["minus"])), 4) if fidelities_by_regime[r]["minus"] else 0.0,
        }

    category_summary: Dict[str, Dict[str, float]] = {}
    for c in ["TP", "FN", "FP", "TN"]:
        category_summary[c] = {
            "mean_fid_plus": round(float(np.mean(fidelities_by_category[c])), 4) if fidelities_by_category[c] else 0.0,
            "mean_fid_minus": round(float(np.mean(fidelities_minus_by_category[c])), 4) if fidelities_minus_by_category[c] else 0.0,
            "count": len(fidelities_by_category[c]),
        }

    overall_fidelity_summary: Dict[str, Dict[str, float]] = {}
    for m in fidelity_results_by_method.keys():
        plus_arr = np.array(fidelity_results_by_method[m])
        minus_arr = np.array(fidelity_minus_by_method[m])
        overall_fidelity_summary[m] = {
            "mean_fidelity_plus": round(float(np.mean(plus_arr)), 4),
            "std_fidelity_plus": round(float(np.std(plus_arr)), 4),
            "median_fidelity_plus": round(float(np.median(plus_arr)), 4),
            "mean_fidelity_minus": round(float(np.mean(minus_arr)), 4),
            "std_fidelity_minus": round(float(np.std(minus_arr)), 4),
            "median_fidelity_minus": round(float(np.median(minus_arr)), 4),
        }

    gat_summary = {
        "is_available": gat_extractor.is_available,
        "mean_spearman_rho": round(float(np.mean(gat_agreement_rhos)), 4) if gat_agreement_rhos else 0.0,
        "mean_top_k_jaccard_overlap": round(float(np.mean(gat_agreement_jaccards)), 4) if gat_agreement_jaccards else 0.0,
        "note": "GAT attention coefficients are model-internal signals and are not assumed to be faithful post-hoc explanations.",
    }

    master_xai_results = {
        "experiment": "EXP-05",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "GraphSAGE",
        "checkpoint": str(sage_ckpt),
        "total_targets_evaluated": len(all_explanations),
        "sampling_strategy": "Stratified across TP, FN, FP, TN and Pre-Shock, Shock, Post-Shock regimes",
        "overall_fidelity_benchmark": overall_fidelity_summary,
        "category_fidelity_benchmark": category_summary,
        "regime_fidelity_benchmark": regime_summary,
        "budget_sparsity_tradeoff": budget_summary_for_plot,
        "stability_metrics": {
            "mean_edge_jaccard": round(float(np.mean(stability_jaccard_scores)), 4) if stability_jaccard_scores else 0.0,
            "std_edge_jaccard": round(float(np.std(stability_jaccard_scores)), 4) if stability_jaccard_scores else 0.0,
            "num_targets_evaluated": len(stability_jaccard_scores),
        },
        "gat_attention_comparison": gat_summary,
        "latency_profile": latency_summary.to_dict(),
    }

    # Save Master JSONs
    fid_json_path = results_dir / "exp05_xai_fidelity.json"
    with open(fid_json_path, "w", encoding="utf-8") as f:
        json.dump(master_xai_results, f, indent=2)
    logger.info(f"Saved EXP-05 master fidelity results to {fid_json_path}")

    summary_json_path = results_dir / "exp05_xai_summary.json"
    summary_dict = {
        "overall_fidelity": overall_fidelity_summary,
        "category_fidelity": category_summary,
        "regime_fidelity": regime_summary,
        "latency_profile": latency_summary.to_dict(),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved EXP-05 summary to {summary_json_path}")

    # 7. Generate Research Plots
    logger.info("\n--- Generating EXP-05 Visualizations ---")
    plot_fidelity_plus_comparison(fidelity_results_by_method, results_dir / "exp05_fidelity_plus.png")
    plot_fidelity_minus_comparison(fidelity_minus_by_method, results_dir / "exp05_fidelity_minus.png")
    plot_sparsity_tradeoff(budget_summary_for_plot, results_dir / "exp05_sparsity.png")
    if stability_jaccard_scores:
        plot_explanation_stability(stability_jaccard_scores, results_dir / "exp05_explanation_stability.png")
    plot_feature_attribution_distribution(all_explanations, results_dir / "exp05_feature_attribution.png")
    plot_edge_attribution_distribution(all_explanations, results_dir / "exp05_edge_attribution.png")
    plot_latency_distribution(latency_profiler.latencies_ms, results_dir / "exp05_latency.png")
    plot_temporal_fidelity(regime_summary, results_dir / "exp05_temporal_fidelity.png")

    # Generate Representative Subgraph Visualizations
    tp_sample = next((e for e in all_explanations if e.category == "TP"), None)
    fn_sample = next((e for e in all_explanations if e.category == "FN"), None)
    fp_sample = next((e for e in all_explanations if e.category == "FP"), None)
    shock_sample = next((e for e in all_explanations if e.period == "shock_period"), None)

    if tp_sample:
        plot_representative_subgraph_explanation(tp_sample, explanations_dir / f"subgraph_TP_{tp_sample.tx_id[:8]}.png")
    if fn_sample:
        plot_representative_subgraph_explanation(fn_sample, explanations_dir / f"subgraph_FN_{fn_sample.tx_id[:8]}.png")
    if fp_sample:
        plot_representative_subgraph_explanation(fp_sample, explanations_dir / f"subgraph_FP_{fp_sample.tx_id[:8]}.png")
    if shock_sample:
        plot_representative_subgraph_explanation(shock_sample, explanations_dir / f"subgraph_SHOCK_{shock_sample.tx_id[:8]}.png")

    # 8. Save Human-Readable Text Report
    txt_report_path = results_dir / "exp05_xai_report.txt"
    lines = [
        "=" * 95,
        "ELLIPTIC BITCOIN DATASET — EXP-05 EXPLAINABILITY & FIDELITY BENCHMARK REPORT",
        "=" * 95,
        f"Generated (UTC)          : {master_xai_results['timestamp_utc']}",
        f"Evaluated Model          : {master_xai_results['model_architecture']} (frozen checkpoint)",
        f"Total Explanations       : {len(all_explanations)} transactions evaluated (TP/FN/FP/TN stratified)",
        "",
        "-" * 95,
        "1. OVERALL FAITHFULNESS & SPARSITY BENCHMARK (10% BUDGET)",
        "-" * 95,
        f"{'Method':<20} | {'Mean Fid+ (Sufficiency)':<25} | {'Mean Fid- (Necessity)':<25} | {'Edge Sparsity'}",
        "-" * 95,
    ]
    for m, m_info in overall_fidelity_summary.items():
        lines.append(
            f"{m.replace('_', ' ').title():<20} | "
            f"{m_info['mean_fidelity_plus']:>+.4f} +/- {m_info['std_fidelity_plus']:.4f}         | "
            f"{m_info['mean_fidelity_minus']:>+.4f} +/- {m_info['std_fidelity_minus']:.4f}         | "
            f"90.0%"
        )
    lines.extend([
        "-" * 95,
        "",
        "-" * 95,
        "2. FIDELITY BY TARGET CATEGORY (TP, FN, FP, TN)",
        "-" * 95,
        f"{'Category':<10} | {'Sample Count':<14} | {'Mean Fid+ (Sufficiency)':<25} | {'Mean Fid- (Necessity)'}",
        "-" * 95,
    ])
    for c, c_info in category_summary.items():
        lines.append(
            f"{c:<10} | {c_info['count']:<14} | {c_info['mean_fid_plus']:>+.4f}                    | {c_info['mean_fid_minus']:>+.4f}"
        )
    lines.extend([
        "-" * 95,
        "",
        "-" * 95,
        "3. FIDELITY ACROSS TEMPORAL REGIMES",
        "-" * 95,
        f"{'Regime':<20} | {'Mean Fid+ (Sufficiency)':<25} | {'Mean Fid- (Necessity)'}",
        "-" * 95,
    ])
    for r, r_info in regime_summary.items():
        lines.append(
            f"{r.replace('_', ' ').title():<20} | {r_info['mean_fid_plus']:>+.4f}                    | {r_info['mean_fid_minus']:>+.4f}"
        )
    lines.extend([
        "-" * 95,
        "",
        "-" * 95,
        "4. GAT MODEL-INTERNAL ATTENTION VS. GNNEXPLAINER RANK AGREEMENT",
        "-" * 95,
        f"GAT Checkpoint Available      : {gat_summary['is_available']}",
        f"Mean Spearman Rank Correlation : rho = {gat_summary['mean_spearman_rho']:+.4f}",
        f"Mean Top-10 Jaccard Overlap    : J = {gat_summary['mean_top_k_jaccard_overlap']:.4f}",
        f"Note                          : {gat_summary['note']}",
        "",
        "-" * 95,
        "5. EXPLANATION STABILITY & LATENCY PROFILING",
        "-" * 95,
        f"Explanation Stability (Jaccard) : {master_xai_results['stability_metrics']['mean_edge_jaccard']:.4f} +/- {master_xai_results['stability_metrics']['std_edge_jaccard']:.4f}",
        f"Mean Generation Latency (ms)    : {latency_summary.mean_latency_ms:.2f} ms",
        f"Median Generation Latency (ms)  : {latency_summary.median_latency_ms:.2f} ms",
        f"P95 Generation Latency (ms)     : {latency_summary.p95_latency_ms:.2f} ms",
        "=" * 95,
    ])
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved human-readable XAI report to {txt_report_path}")

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 95)
    print("PHASE 6: EXP-05 EXPLAINABILITY & FIDELITY BENCHMARK COMPLETE")
    print("=" * 95)
    print(f"Total Targets Evaluated : {len(all_explanations)}")
    print(f"Execution Duration      : {elapsed.total_seconds():.2f} seconds")
    print(f"Mean Generation Latency : {latency_summary.mean_latency_ms:.2f} ms per transaction")
    print("-" * 95)
    print(f"GNNExplainer Mean Fid+  : {overall_fidelity_summary['gnn_explainer']['mean_fidelity_plus']:+.4f} (vs. Random: {overall_fidelity_summary['random']['mean_fidelity_plus']:+.4f})")
    print(f"GNNExplainer Mean Fid-  : {overall_fidelity_summary['gnn_explainer']['mean_fidelity_minus']:+.4f} (vs. Random: {overall_fidelity_summary['random']['mean_fidelity_minus']:+.4f})")
    print(f"GAT Rank Correlation    : rho = {gat_summary['mean_spearman_rho']:+.4f} | Jaccard = {gat_summary['mean_top_k_jaccard_overlap']:.4f}")
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
