"""Training orchestrator and tabular dataset builder for baseline experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader
from src.module4_gnn.evaluator import EvaluationResult, ModelEvaluator
from src.module4_gnn.models.baselines import (
    LightGBMBaseline,
    MLPBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)

logger = get_logger("Module4.Trainer")


class TabularDatasetBuilder:
    """Builds clean supervised training, validation, and test matrices from PyG temporal graphs."""

    def __init__(self, graphs_dir: Optional[Union[str, Path]] = None):
        self.loader = TimestepGraphLoader(graphs_dir)

    def extract_labeled_matrices(
        self,
        feature_config: str = "original_all"
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Loads all 49 timestep graphs and extracts strictly labeled rows (y in {0, 1}).
        
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        logger.info(f"Extracting labeled tabular matrices (feature_config='{feature_config}')...")

        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []

        for ts in range(1, 50):
            data = self.loader.load_graph(ts)
            x_feat = self.loader.get_feature_matrix(data, config=feature_config).numpy()
            y_labels = data.y.numpy()

            # Filter for labeled nodes strictly (y != -1)
            labeled_mask = (y_labels != -1)
            if not labeled_mask.any():
                continue

            x_labeled = x_feat[labeled_mask]
            y_labeled = y_labels[labeled_mask]

            if data.split == "train":
                train_x.append(x_labeled)
                train_y.append(y_labeled)
            elif data.split == "val":
                val_x.append(x_labeled)
                val_y.append(y_labeled)
            elif data.split == "test":
                test_x.append(x_labeled)
                test_y.append(y_labeled)
            else:
                raise ValueError(f"Unknown split '{data.split}' at timestep {ts}")

        X_train = np.vstack(train_x).astype(np.float32)
        y_train = np.concatenate(train_y).astype(np.int64)

        X_val = np.vstack(val_x).astype(np.float32)
        y_val = np.concatenate(val_y).astype(np.int64)

        X_test = np.vstack(test_x).astype(np.float32)
        y_test = np.concatenate(test_y).astype(np.int64)

        logger.info(
            f"Supervised matrices extracted: "
            f"Train={X_train.shape} (Illicit={np.sum(y_train==1):,}, Licit={np.sum(y_train==0):,}), "
            f"Val={X_val.shape} (Illicit={np.sum(y_val==1):,}, Licit={np.sum(y_val==0):,}), "
            f"Test={X_test.shape} (Illicit={np.sum(y_test==1):,}, Licit={np.sum(y_test==0):,})"
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class TabularBaselineTrainer:
    """Trains, calibrates, and evaluates all four tabular baselines for EXP-01."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ):
        self.models_dir = Path(models_dir) if models_dir else Path("models/baselines")
        self.results_dir = Path(results_dir) if results_dir else default_config.paths.processed_data_dir / "experiments"
        self.random_state = random_state
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_exp01(
        self,
        feature_config: str = "original_all"
    ) -> Dict:
        """
        Executes EXP-01:
        1. Builds supervised matrices.
        2. Trains Random Forest, XGBoost, LightGBM, and MLP.
        3. Calibrates threshold tau* on validation set.
        4. Evaluates frozen threshold on out-of-time test set.
        5. Saves model checkpoints and experiment JSON.
        """
        builder = TabularDatasetBuilder()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = builder.extract_labeled_matrices(feature_config)

        # Imbalance ratio from training data strictly
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        scale_pos_weight = float(num_neg / num_pos)

        in_features = X_train.shape[1]

        models = {
            "random_forest": RandomForestBaseline(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "xgboost": XGBoostBaseline(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
            ),
            "lightgbm": LightGBMBaseline(
                n_estimators=150,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
            ),
            "mlp": MLPBaseline(
                in_features=in_features,
                hidden_dim1=128,
                hidden_dim2=64,
                dropout=0.3,
                lr=0.001,
                random_state=self.random_state,
            ),
        }

        results_data = {
            "experiment": "EXP-01",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_configuration": feature_config,
            "feature_count": in_features,
            "train_timesteps": [1, 34],
            "validation_timesteps": [35, 39],
            "test_timesteps": [40, 49],
            "dataset_counts": {
                "train_samples": len(y_train),
                "train_illicit": int(num_pos),
                "train_licit": int(num_neg),
                "train_imbalance_ratio": round(scale_pos_weight, 2),
                "val_samples": len(y_val),
                "val_illicit": int(np.sum(y_val == 1)),
                "val_licit": int(np.sum(y_val == 0)),
                "test_samples": len(y_test),
                "test_illicit": int(np.sum(y_test == 1)),
                "test_licit": int(np.sum(y_test == 0)),
            },
            "models": {},
            "raw_predictions": {},
        }

        evaluator = ModelEvaluator()

        for model_name, model in models.items():
            logger.info(f"\n{'='*30} Training {model_name.upper()} {'='*30}")
            t0 = time.time()

            # Train
            if model_name in ["xgboost", "lightgbm", "mlp"]:
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)

            train_duration = round(time.time() - t0, 2)

            # Predict probabilities
            val_probs = model.predict_proba(X_val)
            test_probs = model.predict_proba(X_test)

            # Calibrate threshold on validation
            tau_star, val_best_f1 = evaluator.calibrate_threshold(y_val, val_probs)

            # Validation evaluation
            val_metrics = evaluator.evaluate(y_val, val_probs, threshold=tau_star)
            # Test evaluation with frozen threshold
            test_metrics = evaluator.evaluate(y_test, test_probs, threshold=tau_star)

            # Save model artifact
            if model_name == "random_forest":
                model.save(self.models_dir / "random_forest.joblib")
            elif model_name == "xgboost":
                model.save(self.models_dir / "xgboost.json")
            elif model_name == "lightgbm":
                model.save(self.models_dir / "lightgbm.txt")
            elif model_name == "mlp":
                model.save(self.models_dir / "mlp.pt")

            results_data["models"][model_name] = {
                "training_duration_seconds": train_duration,
                "calibrated_threshold": tau_star,
                "validation": val_metrics.to_dict(),
                "test": test_metrics.to_dict(),
            }

            results_data["raw_predictions"][model_name] = {
                "val_probs": val_probs.tolist(),
                "test_probs": test_probs.tolist(),
            }

            logger.info(
                f"{model_name.upper()} Test Results (tau*={tau_star:.4f}): "
                f"Illicit F1={test_metrics.f1_illicit:.4f}, "
                f"PR-AUC={test_metrics.pr_auc:.4f}, "
                f"Precision={test_metrics.precision_illicit:.4f}, "
                f"Recall={test_metrics.recall_illicit:.4f}, "
                f"ROC-AUC={test_metrics.roc_auc:.4f}"
            )

        # Store test labels for plotting
        results_data["test_labels"] = y_test.tolist()
        results_data["val_labels"] = y_val.tolist()

        # Save JSON results (without heavy raw predictions in main summary)
        summary_data = {k: v for k, v in results_data.items() if k not in ["raw_predictions", "test_labels", "val_labels"]}
        json_path = self.results_dir / "exp01_tabular_baselines.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved EXP-01 results to {json_path}")

        return results_data
