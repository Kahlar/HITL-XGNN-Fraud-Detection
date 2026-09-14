"""Phase 3 Master Execution Script: Train, calibrate, and evaluate EXP-01 tabular baselines."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import default_config
from src.common.logger import get_logger
from src.module4_gnn.trainer import TabularBaselineTrainer
from src.module4_gnn.visualization import (
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

logger = get_logger("Phase3.Runner")


def main() -> int:
    """Executes the complete Phase 3 EXP-01 baseline benchmark."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 3: TABULAR BASELINES (MODULE 4 / EXP-01)")
    logger.info("=" * 80)

    models_dir = Path("models/baselines")
    results_dir = default_config.paths.processed_data_dir / "experiments"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run EXP-01 Baseline Models (RF, XGBoost, LightGBM, MLP)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 1: EXECUTING EXP-01 BENCHMARK (165 ORIGINAL FEATURES) ---")
    trainer = TabularBaselineTrainer(
        models_dir=models_dir,
        results_dir=results_dir,
        random_state=default_config.random_seed,
    )

    exp_results = trainer.run_exp01(feature_config="original_all")

    # -------------------------------------------------------------------------
    # Step 2: Generate High-Resolution Visualization Curves
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 2: GENERATING PR, ROC, AND COMPARISON CURVES ---")
    y_test = np.array(exp_results["test_labels"], dtype=int)
    test_predictions = {
        model_name: np.array(exp_results["raw_predictions"][model_name]["test_probs"], dtype=np.float32)
        for model_name in exp_results["models"].keys()
    }

    pr_curve_path = results_dir / "exp01_pr_curves.png"
    roc_curve_path = results_dir / "exp01_roc_curves.png"
    comp_plot_path = results_dir / "exp01_model_comparison.png"

    plot_pr_curves(y_test, test_predictions, save_path=pr_curve_path)
    plot_roc_curves(y_test, test_predictions, save_path=roc_curve_path)
    plot_model_comparison(exp_results["models"], save_path=comp_plot_path)

    elapsed = datetime.now() - start_time

    # -------------------------------------------------------------------------
    # Final Tabular Comparison Summary Table
    # -------------------------------------------------------------------------
    counts = exp_results["dataset_counts"]
    print("\n" + "=" * 95)
    print("PHASE 3: EXP-01 TABULAR BASELINES BENCHMARK RESULTS")
    print("=" * 95)
    print(f"Status                 : SUCCESS")
    print(f"Execution Duration     : {elapsed.total_seconds():.2f} seconds")
    print(f"Feature Set            : original_all (165 features)")
    print(f"Train Dataset (t=1..34): {counts['train_samples']:,} labeled ({counts['train_illicit']:,} illicit, {counts['train_licit']:,} licit | Imbalance: {counts['train_imbalance_ratio']:.2f}:1)")
    print(f"Val Dataset (t=35..39) : {counts['val_samples']:,} labeled ({counts['val_illicit']:,} illicit, {counts['val_licit']:,} licit)")
    print(f"Test Dataset (t=40..49): {counts['test_samples']:,} labeled ({counts['test_illicit']:,} illicit, {counts['test_licit']:,} licit)")
    print("-" * 95)
    print(f"{'Model':<16} | {'tau*':<6} | {'Val F1':<7} | {'Test F1':<7} | {'Test PR-AUC':<11} | {'Precision':<9} | {'Recall':<7} | {'ROC-AUC':<7} | {'Runtime'}")
    print("-" * 95)

    for m_name, m_data in exp_results["models"].items():
        v = m_data["validation"]
        t = m_data["test"]
        tau = m_data["calibrated_threshold"]
        dur = m_data["training_duration_seconds"]
        print(
            f"{m_name:<16} | {tau:<6.3f} | {v['f1_illicit']:<7.4f} | {t['f1_illicit']:<7.4f} | "
            f"{t['pr_auc']:<11.4f} | {t['precision_illicit']:<9.4f} | {t['recall_illicit']:<7.4f} | {t['roc_auc']:<7.4f} | {dur:>5.2f}s"
        )

    print("-" * 95)
    print("Generated Artifacts:")
    print(f"  - Model Artifacts   : {models_dir}")
    print(f"  - EXP-01 JSON       : {results_dir / 'exp01_tabular_baselines.json'}")
    print(f"  - PR Curves Plot    : {pr_curve_path}")
    print(f"  - ROC Curves Plot   : {roc_curve_path}")
    print(f"  - Comparison Bar    : {comp_plot_path}")
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
