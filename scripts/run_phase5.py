"""Phase 5 Master Execution Script: EXP-06 Temporal Robustness & Distribution Shift Benchmark."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import default_config
from src.common.logger import get_logger
from src.module5_temporal_drift.drift_evaluator import TemporalDriftEvaluator
from src.module5_temporal_drift.drift_visualization import (
    plot_drift_summary_dashboard,
    plot_f1_by_timestep,
    plot_illicit_prevalence,
    plot_pr_auc_by_timestep,
    plot_precision_recall_by_timestep,
)

logger = get_logger("Phase5.Runner")


def main() -> int:
    """Executes the complete Phase 5 EXP-06 temporal drift benchmark."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 5: TEMPORAL DRIFT & DISTRIBUTION SHIFT BENCHMARK (EXP-06)")
    logger.info("=" * 80)

    graphs_dir = default_config.paths.processed_data_dir / "graphs"
    models_dir = Path("models/gnn")
    results_dir = default_config.paths.processed_data_dir / "experiments"

    # Initialize Temporal Drift Evaluator
    evaluator = TemporalDriftEvaluator(
        graphs_dir=graphs_dir,
        models_dir=models_dir,
        results_dir=results_dir,
    )

    # Execute EXP-06 evaluation on frozen GraphSAGE and XGBoost models
    drift_results = evaluator.run_exp06(
        graphsage_checkpoint=models_dir / "graphsage_best.pt",
        xgboost_checkpoint=Path("models/baselines/xgboost.json"),
        threshold=0.5517,      # Frozen from validation
        xgb_threshold=0.9014,  # Frozen from validation
    )

    # -------------------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------------------
    logger.info("\n--- GENERATING EXP-06 TEMPORAL DRIFT PLOTS ---")
    metrics = drift_results["per_timestep_metrics"]
    xgb_metrics = drift_results.get("xgboost_per_timestep_metrics")

    f1_plot_path = results_dir / "exp06_f1_by_timestep.png"
    prauc_plot_path = results_dir / "exp06_pr_auc_by_timestep.png"
    prec_rec_plot_path = results_dir / "exp06_precision_recall_by_timestep.png"
    prev_plot_path = results_dir / "exp06_illicit_prevalence.png"
    dashboard_plot_path = results_dir / "exp06_drift_summary.png"

    plot_f1_by_timestep(metrics, xgb_metrics, save_path=f1_plot_path)
    plot_pr_auc_by_timestep(metrics, xgb_metrics, save_path=prauc_plot_path)
    plot_precision_recall_by_timestep(metrics, save_path=prec_rec_plot_path)
    plot_illicit_prevalence(metrics, save_path=prev_plot_path)
    plot_drift_summary_dashboard(drift_results, save_path=dashboard_plot_path)

    elapsed = datetime.now() - start_time

    # -------------------------------------------------------------------------
    # Final Benchmark Summary Display
    # -------------------------------------------------------------------------
    p_data = drift_results["period_analysis"]
    stats = drift_results["statistical_summary"]
    ext = drift_results["extrema_analysis"]

    print("\n" + "=" * 105)
    print("PHASE 5: EXP-06 TEMPORAL DRIFT & DISTRIBUTION SHIFT BENCHMARK RESULTS")
    print("=" * 105)
    print(f"Status                 : SUCCESS")
    print(f"Execution Duration     : {elapsed.total_seconds():.2f} seconds")
    print(f"Evaluated Model        : GraphSAGE (Frozen Checkpoint: models/gnn/graphsage_best.pt)")
    print(f"Frozen Threshold       : tau* = 0.5517 (strictly from validation)")
    print(f"Test Timesteps         : t=40..49 (10 discrete temporal subgraphs, 11,184 labeled nodes)")
    print("-" * 105)
    print(f"{'TS':<4} | {'Period':<13} | {'Labeled':<7} | {'Illicit':<7} | {'Prevalence':<10} | {'F1 (Sage)':<9} | {'PR-AUC':<7} | {'Prec':<7} | {'Recall':<7} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 105)

    for m in metrics:
        print(
            f"{m['timestep']:<4} | {m['period']:<13} | {m['labeled_nodes']:<7} | {m['illicit_count']:<7} | "
            f"{m['illicit_prevalence']*100:>5.2f}%    | {m['f1_illicit']:<9.4f} | {m['pr_auc']:<7.4f} | "
            f"{m['precision_illicit']:<7.4f} | {m['recall_illicit']:<7.4f} | {m['true_positives']:<4} | "
            f"{m['false_positives']:<4} | {m['false_negatives']:<4}"
        )

    print("-" * 105)
    print("TEMPORAL REGIME AGGREGATES:")
    print(f"  - Pre-Shock   (t=40..42): Mean F1 = {p_data['pre_shock']['mean_f1']:.4f} | Mean PR-AUC = {p_data['pre_shock']['mean_pr_auc']:.4f} | Mean Prev = {p_data['pre_shock']['mean_illicit_prevalence']*100:.2f}%")
    print(f"  - Shock Window(t=43..46): Mean F1 = {p_data['shock_period']['mean_f1']:.4f} | Mean PR-AUC = {p_data['shock_period']['mean_pr_auc']:.4f} | Mean Prev = {p_data['shock_period']['mean_illicit_prevalence']*100:.2f}% (Acute Drop)")
    print(f"  - Post-Shock  (t=47..49): Mean F1 = {p_data['post_shock']['mean_f1']:.4f} | Mean PR-AUC = {p_data['post_shock']['mean_pr_auc']:.4f} | Mean Prev = {p_data['post_shock']['mean_illicit_prevalence']*100:.2f}%")
    print("-" * 105)
    print("STATISTICAL DRIFT SUMMARY:")
    print(f"  - F1-Score Mean +/- Std  : {stats['f1']['mean']:.4f} +/- {stats['f1']['std']:.4f} (CV = {stats['f1']['cv']:.4f}, Range: [{stats['f1']['min']:.4f}, {stats['f1']['max']:.4f}])")
    print(f"  - PR-AUC Mean +/- Std    : {stats['pr_auc']['mean']:.4f} +/- {stats['pr_auc']['std']:.4f} (CV = {stats['pr_auc']['cv']:.4f}, Range: [{stats['pr_auc']['min']:.4f}, {stats['pr_auc']['max']:.4f}])")
    print(f"  - Largest Consecutive Drop: TS {ext['largest_consecutive_f1_drop']['from_timestep']} -> TS {ext['largest_consecutive_f1_drop']['to_timestep']} (F1 Drop: -{ext['largest_consecutive_f1_drop']['drop_magnitude']:.4f})")
    print("-" * 105)
    print("Generated Artifacts:")
    print(f"  - EXP-06 Drift JSON     : {results_dir / 'exp06_temporal_drift.json'}")
    print(f"  - EXP-06 Summary JSON   : {results_dir / 'exp06_temporal_summary.json'}")
    print(f"  - Human-Readable Report : {results_dir / 'exp06_temporal_drift_report.txt'}")
    print(f"  - F1 Trend Plot         : {f1_plot_path}")
    print(f"  - PR-AUC Trend Plot     : {prauc_plot_path}")
    print(f"  - Prec vs Rec Plot      : {prec_rec_plot_path}")
    print(f"  - Prevalence Plot       : {prev_plot_path}")
    print(f"  - Dashboard Plot        : {dashboard_plot_path}")
    print("=" * 105 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
