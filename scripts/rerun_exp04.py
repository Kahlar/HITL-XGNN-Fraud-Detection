"""Targeted runner for corrected EXP-04: Class Imbalance Loss Comparison on GraphSAGE + original_all (165)."""

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
from src.module4_gnn.gnn_trainer import GNNTrainer
from src.module4_gnn.visualization import plot_loss_comparison

logger = get_logger("Phase4.RerunEXP04")


def main() -> int:
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("RERUNNING EXP-04: LOSS COMPARISON (GraphSAGE on original_all 165 features)")
    logger.info("=" * 80)

    graphs_dir = default_config.paths.processed_data_dir / "graphs"
    models_dir = Path("models/gnn")
    results_dir = default_config.paths.processed_data_dir / "experiments"

    trainer = GNNTrainer(
        graphs_dir=graphs_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        random_state=default_config.random_seed,
    )

    # Run EXP-04 on original_all (165)
    exp04_results = trainer.run_exp04_loss_comparison(
        best_architecture="graphsage",
        feature_config="original_all",
    )

    # Generate updated plot
    plot_loss_comparison(
        exp04_results,
        save_path=results_dir / "exp04_loss_comparison.png",
        title="EXP-04: Class Imbalance Loss Comparison (GraphSAGE, original_all: 165 features)",
    )

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 90)
    print("CORRECTED EXP-04 LOSS COMPARISON RESULTS (GraphSAGE + original_all: 165 features)")
    print("=" * 90)
    print(f"Selected Loss by Validation : {exp04_results['selected_loss_by_validation'].upper()}")
    print(f"Execution Duration          : {elapsed.total_seconds():.2f} seconds")
    print("-" * 90)
    print(f"{'Loss Function':<25} | {'tau*':<6} | {'Val PR-AUC':<10} | {'Val F1':<8} | {'Test F1':<8} | {'Test PR-AUC':<11} | {'Precision':<9} | {'Recall'}")
    print("-" * 90)

    for l_name, l_info in exp04_results["loss_functions"].items():
        v = l_info["validation"]
        t = l_info["test"]
        tau = l_info["calibrated_threshold"]
        print(
            f"{l_name:<25} | {tau:<6.3f} | {v['pr_auc']:<10.4f} | {v['f1_illicit']:<8.4f} | "
            f"{t['f1_illicit']:<8.4f} | {t['pr_auc']:<11.4f} | {t['precision_illicit']:<9.4f} | {t['recall_illicit']:.4f}"
        )
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
