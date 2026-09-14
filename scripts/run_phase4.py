"""Phase 4 Master Execution Script: Train, evaluate, and benchmark GNNs (EXP-02, EXP-03, EXP-04)."""

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
from src.module4_gnn.gnn_trainer import GNNTrainer
from src.module4_gnn.visualization import (
    plot_feature_ablation_comparison,
    plot_full_model_comparison,
    plot_gnn_training_curves,
    plot_loss_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

logger = get_logger("Phase4.Runner")


def main() -> int:
    """Executes the complete Phase 4 GNN benchmark (EXP-02, EXP-03, EXP-04)."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 4: GRAPH NEURAL NETWORKS (EXP-02, EXP-03, EXP-04)")
    logger.info("=" * 80)

    graphs_dir = default_config.paths.processed_data_dir / "graphs"
    models_dir = Path("models/gnn")
    results_dir = default_config.paths.processed_data_dir / "experiments"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GNN Trainer
    trainer = GNNTrainer(
        graphs_dir=graphs_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        random_state=default_config.random_seed,
    )

    # -------------------------------------------------------------------------
    # STEP 1: EXP-02 — GNN Architecture Comparison (165 features)
    # -------------------------------------------------------------------------
    exp02_results = trainer.run_exp02_architecture_comparison(
        feature_config="original_all",
        loss_type="focal",
    )

    # Select best architecture based strictly on validation PR-AUC
    best_arch = max(
        exp02_results["models"].keys(),
        key=lambda m: exp02_results["models"][m]["validation"]["pr_auc"],
    )
    logger.info(f"\n>>> Best GNN Architecture by Validation PR-AUC: {best_arch.upper()} <<<")

    # -------------------------------------------------------------------------
    # STEP 2: EXP-03 — Feature Configuration Ablation
    # -------------------------------------------------------------------------
    exp03_results = trainer.run_exp03_feature_ablation(
        best_architecture=best_arch,
        loss_type="focal",
    )

    # -------------------------------------------------------------------------
    # STEP 3: EXP-04 — Class Imbalance Loss Comparison (Canonical: original_all 165)
    # -------------------------------------------------------------------------
    exp04_results = trainer.run_exp04_loss_comparison(
        best_architecture=best_arch,
        feature_config="original_all",
    )
    best_loss = exp04_results["selected_loss_by_validation"]

    # -------------------------------------------------------------------------
    # STEP 4: Generate All Visualization Artifacts
    # -------------------------------------------------------------------------
    logger.info("\n--- GENERATING PHASE 4 PLOTS AND CURVES ---")
    y_test = np.array(exp02_results["test_targets"], dtype=int)
    gnn_test_preds = {
        name: np.array(exp02_results["raw_predictions"][name]["test_probs"], dtype=np.float32)
        for name in exp02_results["models"].keys()
    }

    # EXP-02 PR & ROC curves
    plot_pr_curves(
        y_test,
        gnn_test_preds,
        save_path=results_dir / "exp02_pr_curves.png",
        title="EXP-02: GNN Architecture Precision-Recall Curves (Test Set: t=40..49)",
    )
    plot_roc_curves(
        y_test,
        gnn_test_preds,
        save_path=results_dir / "exp02_roc_curves.png",
        title="EXP-02: GNN Architecture ROC Curves (Test Set: t=40..49)",
    )

    # GNN Training convergence curves
    plot_gnn_training_curves(
        exp02_results["training_histories"],
        save_path=results_dir / "gnn_training_curves.png",
    )

    # Feature ablation & loss comparison charts
    plot_feature_ablation_comparison(
        exp03_results,
        save_path=results_dir / "exp03_feature_ablation.png",
    )
    plot_loss_comparison(
        exp04_results,
        save_path=results_dir / "exp04_loss_comparison.png",
    )

    # -------------------------------------------------------------------------
    # STEP 5: Full Benchmark Comparison with Phase 3 Baselines
    # -------------------------------------------------------------------------
    logger.info("\n--- COMPILING FULL MODEL BENCHMARK (BASELINES + GNNS) ---")
    exp01_path = results_dir / "exp01_tabular_baselines.json"
    with open(exp01_path, "r", encoding="utf-8") as f:
        exp01_data = json.load(f)

    full_comparison = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_timesteps": [1, 34],
        "validation_timesteps": [35, 39],
        "test_timesteps": [40, 49],
        "models": [],
    }

    # Add tabular baselines from Phase 3
    for b_name, b_info in exp01_data["models"].items():
        t = b_info["test"]
        full_comparison["models"].append({
            "name": b_name.replace("_", " ").title(),
            "type": "Tabular Baseline",
            "is_graph": False,
            "features": "original_all (165)",
            "calibrated_threshold": round(b_info["calibrated_threshold"], 4),
            "f1_illicit": t["f1_illicit"],
            "pr_auc": t["pr_auc"],
            "precision": t["precision_illicit"],
            "recall": t["recall_illicit"],
            "roc_auc": t["roc_auc"],
            "runtime_seconds": b_info["training_duration_seconds"],
        })

    # Add GNN models from Phase 4
    for g_name, g_info in exp02_results["models"].items():
        t = g_info["test"]
        full_comparison["models"].append({
            "name": g_name.upper(),
            "type": "Graph Neural Network",
            "is_graph": True,
            "features": "original_all (165)",
            "calibrated_threshold": round(g_info["calibrated_threshold"], 4),
            "f1_illicit": t["f1_illicit"],
            "pr_auc": t["pr_auc"],
            "precision": t["precision_illicit"],
            "recall": t["recall_illicit"],
            "roc_auc": t["roc_auc"],
            "runtime_seconds": g_info["training_duration_seconds"],
            "parameters": g_info["parameter_count"],
        })

    full_comp_path = results_dir / "full_model_comparison.json"
    with open(full_comp_path, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, indent=2)
    logger.info(f"Saved full model comparison to {full_comp_path}")

    # Plot full benchmark comparison
    plot_full_model_comparison(
        full_comparison["models"],
        save_path=results_dir / "full_model_comparison.png",
    )

    elapsed = datetime.now() - start_time

    # -------------------------------------------------------------------------
    # Final Benchmark Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("PHASE 4: FULL FRAUD DETECTION BENCHMARK (TABULAR BASELINES VS. GRAPH NEURAL NETWORKS)")
    print("=" * 105)
    print(f"Execution Duration : {elapsed.total_seconds():.2f} seconds")
    print(f"Best GNN Model     : {best_arch.upper()} (Selected on Validation PR-AUC)")
    print(f"Feature Set        : original_all (165 features)")
    print(f"Best Loss Function : {best_loss} (EXP-04 Validation Selection)")
    print("-" * 105)
    print(f"{'Model':<18} | {'Type':<12} | {'tau*':<6} | {'Test F1':<7} | {'PR-AUC':<7} | {'Precision':<9} | {'Recall':<7} | {'ROC-AUC':<7} | {'Runtime'}")
    print("-" * 105)

    for m in full_comparison["models"]:
        print(
            f"{m['name']:<18} | {m['type']:<12} | {m['calibrated_threshold']:<6.3f} | {m['f1_illicit']:<7.4f} | "
            f"{m['pr_auc']:<7.4f} | {m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['roc_auc']:<7.4f} | {m['runtime_seconds']:>5.2f}s"
        )

    print("-" * 105)
    print("Generated Artifacts:")
    print(f"  - GNN Checkpoints       : {models_dir}")
    print(f"  - EXP-02 Results JSON   : {results_dir / 'exp02_gnn_comparison.json'}")
    print(f"  - EXP-03 Results JSON   : {results_dir / 'exp03_feature_ablation.json'}")
    print(f"  - EXP-04 Results JSON   : {results_dir / 'exp04_loss_comparison.json'}")
    print(f"  - Full Comparison JSON  : {full_comp_path}")
    print(f"  - Full Comparison Plot  : {results_dir / 'full_model_comparison.png'}")
    print(f"  - GNN Training Curves   : {results_dir / 'gnn_training_curves.png'}")
    print(f"  - Feature Ablation Plot : {results_dir / 'exp03_feature_ablation.png'}")
    print(f"  - Loss Comparison Plot  : {results_dir / 'exp04_loss_comparison.png'}")
    print("=" * 105 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
