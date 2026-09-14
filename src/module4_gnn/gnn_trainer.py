"""GNN Training engine and experiment orchestrator for EXP-02, EXP-03, and EXP-04."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader
from src.module4_gnn.evaluator import EvaluationResult, ModelEvaluator
from src.module4_gnn.loss import FocalLoss, WeightedCrossEntropyLoss
from src.module4_gnn.models.gat import GATNet
from src.module4_gnn.models.gcn import GCNNet
from src.module4_gnn.models.graphsage import GraphSAGENet

logger = get_logger("Module4.GNNTrainer")


class GNNTrainer:
    """Trains, evaluates, and benchmarks Graph Neural Networks on discrete temporal subgraphs."""

    def __init__(
        self,
        graphs_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.graphs_dir = Path(graphs_dir) if graphs_dir else default_config.paths.processed_data_dir / "graphs"
        self.models_dir = Path(models_dir) if models_dir else Path("models/gnn")
        self.results_dir = Path(results_dir) if results_dir else default_config.paths.processed_data_dir / "experiments"
        self.loader = TimestepGraphLoader(self.graphs_dir)
        self.random_state = random_state

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Preload all 49 graph objects in memory (only ~138 MB) for high-speed training
        logger.info(f"Preloading 49 temporal graphs on {self.device}...")
        self.graphs: Dict[int, Data] = {}
        for ts in range(1, 50):
            self.graphs[ts] = self.loader.load_graph(ts)

        # Compute training class imbalance strictly from training timesteps (t=1..34)
        train_pos, train_neg = 0, 0
        for ts in range(1, 35):
            y_labels = self.graphs[ts].y.numpy()
            train_pos += int(np.sum(y_labels == 1))
            train_neg += int(np.sum(y_labels == 0))

        self.scale_pos_weight = float(train_neg / train_pos)
        logger.info(
            f"Training Imbalance: Licit={train_neg:,}, Illicit={train_pos:,}, "
            f"scale_pos_weight={self.scale_pos_weight:.2f}"
        )

    def _get_loss_fn(self, loss_type: str) -> nn.Module:
        """Instantiates the specified loss function with weights derived strictly from training data."""
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "weighted_cross_entropy":
            weights = torch.tensor([1.0, self.scale_pos_weight], dtype=torch.float32).to(self.device)
            return WeightedCrossEntropyLoss(weight=weights)
        elif loss_type == "focal":
            return FocalLoss(alpha=0.75, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def train_model(
        self,
        model: nn.Module,
        feature_config: str = "original_all",
        loss_type: str = "focal",
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        max_epochs: int = 40,
        patience: int = 15,
    ) -> Tuple[nn.Module, Dict[str, List[float]], int]:
        """
        Trains a GNN across temporal training graphs (t=1..34) with validation early stopping (t=35..39).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        model = model.to(self.device)
        loss_fn = self._get_loss_fn(loss_type)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_f1": [],
        }

        best_val_score = -1.0
        best_epoch = 1
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            total_train_nodes = 0

            # Iterate through temporal training graphs (t=1..34)
            for ts in range(1, 35):
                data = self.graphs[ts]
                x = self.loader.get_feature_matrix(data, config=feature_config).to(self.device)
                edge_index = data.edge_index.to(self.device)
                y = data.y.to(self.device)

                # Loss strictly over labeled nodes (y != -1)
                labeled_mask = (y != -1)
                if not labeled_mask.any():
                    continue

                optimizer.zero_grad()
                logits = model(x, edge_index)
                loss = loss_fn(logits[labeled_mask], y[labeled_mask])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * int(labeled_mask.sum().item())
                total_train_nodes += int(labeled_mask.sum().item())

            avg_train_loss = epoch_loss / max(total_train_nodes, 1)
            history["train_loss"].append(round(avg_train_loss, 4))

            # Validation evaluation across t=35..39
            val_probs, val_targets = self.predict_timesteps(model, timesteps=range(35, 40), feature_config=feature_config)
            
            # Compute Val PR-AUC & F1
            prec, rec, _ = precision_recall_curve(val_targets, val_probs, pos_label=1)
            val_pr_auc = float(auc(rec, prec))
            
            # Quick threshold calibration for val F1 monitoring
            _, val_f1 = ModelEvaluator.calibrate_threshold(val_targets, val_probs, num_thresholds=50)

            history["val_pr_auc"].append(round(val_pr_auc, 4))
            history["val_f1"].append(round(val_f1, 4))

            if val_pr_auc > best_val_score:
                best_val_score = val_pr_auc
                best_epoch = epoch
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best Val PR-AUC: {best_val_score:.4f} (Epoch {best_epoch})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history, best_epoch

    def predict_timesteps(
        self,
        model: nn.Module,
        timesteps: range,
        feature_config: str = "original_all",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on given timesteps, returning concatenated probabilities and labels for labeled nodes.
        """
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for ts in timesteps:
                data = self.graphs[ts]
                x = self.loader.get_feature_matrix(data, config=feature_config).to(self.device)
                edge_index = data.edge_index.to(self.device)
                y = data.y.numpy()

                labeled_mask = (y != -1)
                if not labeled_mask.any():
                    continue

                logits = model(x, edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_probs.append(probs[labeled_mask])
                all_targets.append(y[labeled_mask])

        return np.concatenate(all_probs).astype(np.float32), np.concatenate(all_targets).astype(np.int64)

    def run_exp02_architecture_comparison(
        self,
        feature_config: str = "original_all",
        loss_type: str = "focal",
    ) -> Dict:
        """
        Executes EXP-02: GNN Architecture Comparison (GCN, GraphSAGE, Multi-Head GAT).
        """
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTING EXP-02: GNN ARCHITECTURE COMPARISON (GCN, GraphSAGE, GAT)")
        logger.info("=" * 80)

        in_dim = self.loader.get_feature_matrix(self.graphs[1], config=feature_config).size(1)

        model_factories = {
            "gcn": lambda: GCNNet(in_features=in_dim, hidden_dim=128, dropout=0.3),
            "graphsage": lambda: GraphSAGENet(in_features=in_dim, hidden_dim=128, dropout=0.3, aggr="mean"),
            "gat": lambda: GATNet(in_features=in_dim, hidden_dim=32, heads=4, dropout=0.3),
        }

        results = {
            "experiment": "EXP-02",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_configuration": feature_config,
            "feature_count": in_dim,
            "loss_function": loss_type,
            "train_timesteps": [1, 34],
            "validation_timesteps": [35, 39],
            "test_timesteps": [40, 49],
            "models": {},
            "raw_predictions": {},
            "training_histories": {},
        }

        evaluator = ModelEvaluator()

        for name, factory in model_factories.items():
            logger.info(f"\n{'='*25} Training {name.upper()} {'='*25}")
            model = factory()
            param_count = model.count_parameters()

            t0 = time.time()
            trained_model, history, best_epoch = self.train_model(
                model=model,
                feature_config=feature_config,
                loss_type=loss_type,
                max_epochs=40,
                patience=15,
            )
            train_duration = round(time.time() - t0, 2)

            # Inference & timing
            t_inf_start = time.time()
            val_probs, val_targets = self.predict_timesteps(trained_model, range(35, 40), feature_config)
            test_probs, test_targets = self.predict_timesteps(trained_model, range(40, 50), feature_config)
            inf_duration = round(time.time() - t_inf_start, 3)

            # Calibrate threshold on validation
            tau_star, val_best_f1 = evaluator.calibrate_threshold(val_targets, val_probs)

            # Evaluate with frozen threshold
            val_metrics = evaluator.evaluate(val_targets, val_probs, threshold=tau_star)
            test_metrics = evaluator.evaluate(test_targets, test_probs, threshold=tau_star)

            # Save model checkpoint
            ckpt_path = self.models_dir / f"{name}_best.pt"
            trained_model.save_checkpoint(
                ckpt_path,
                metadata={
                    "architecture": name,
                    "in_features": in_dim,
                    "best_epoch": best_epoch,
                    "calibrated_threshold": tau_star,
                    "validation_metrics": val_metrics.to_dict(),
                    "test_metrics": test_metrics.to_dict(),
                },
            )

            results["models"][name] = {
                "parameter_count": param_count,
                "best_epoch": best_epoch,
                "training_duration_seconds": train_duration,
                "inference_duration_seconds": inf_duration,
                "calibrated_threshold": tau_star,
                "validation": val_metrics.to_dict(),
                "test": test_metrics.to_dict(),
            }
            results["raw_predictions"][name] = {
                "val_probs": val_probs.tolist(),
                "test_probs": test_probs.tolist(),
            }
            results["training_histories"][name] = history

            logger.info(
                f"{name.upper()} Test Results (tau*={tau_star:.4f}): "
                f"Illicit F1={test_metrics.f1_illicit:.4f}, "
                f"PR-AUC={test_metrics.pr_auc:.4f}, "
                f"Precision={test_metrics.precision_illicit:.4f}, "
                f"Recall={test_metrics.recall_illicit:.4f}, "
                f"ROC-AUC={test_metrics.roc_auc:.4f}, "
                f"Params={param_count:,}"
            )

        # Store test and val targets for plotting
        results["test_targets"] = test_targets.tolist()
        results["val_targets"] = val_targets.tolist()

        # Save EXP-02 JSON
        json_summary = {k: v for k, v in results.items() if k not in ["raw_predictions", "test_targets", "val_targets"]}
        json_path = self.results_dir / "exp02_gnn_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2)
        logger.info(f"Saved EXP-02 results to {json_path}")

        return results

    def run_exp03_feature_ablation(
        self,
        best_architecture: str = "graphsage",
        loss_type: str = "focal",
    ) -> Dict:
        """
        Executes EXP-03: Feature Configuration Ablation on best GNN architecture.
        Compares: original_all (165), original_local (93), engineered (5), combined (170).
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"EXECUTING EXP-03: FEATURE ABLATION (Architecture: {best_architecture.upper()})")
        logger.info("=" * 80)

        feature_configs = ["original_all", "original_local", "engineered", "combined"]
        results = {
            "experiment": "EXP-03",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "architecture": best_architecture,
            "loss_function": loss_type,
            "configurations": {},
        }

        evaluator = ModelEvaluator()
        best_val_prauc = -1.0
        selected_feature_config = "original_all"

        for f_cfg in feature_configs:
            in_dim = self.loader.get_feature_matrix(self.graphs[1], config=f_cfg).size(1)
            logger.info(f"\n--- Evaluating Feature Configuration: {f_cfg} (dim={in_dim}) ---")

            if best_architecture == "graphsage":
                model = GraphSAGENet(in_features=in_dim, hidden_dim=128, dropout=0.3, aggr="mean")
            elif best_architecture == "gcn":
                model = GCNNet(in_features=in_dim, hidden_dim=128, dropout=0.3)
            elif best_architecture == "gat":
                model = GATNet(in_features=in_dim, hidden_dim=32, heads=4, dropout=0.3)
            else:
                raise ValueError(f"Unknown architecture: {best_architecture}")

            t0 = time.time()
            trained_model, history, best_epoch = self.train_model(
                model=model,
                feature_config=f_cfg,
                loss_type=loss_type,
                max_epochs=40,
                patience=15,
            )
            duration = round(time.time() - t0, 2)

            val_probs, val_targets = self.predict_timesteps(trained_model, range(35, 40), f_cfg)
            test_probs, test_targets = self.predict_timesteps(trained_model, range(40, 50), f_cfg)

            tau_star, val_best_f1 = evaluator.calibrate_threshold(val_targets, val_probs)
            val_metrics = evaluator.evaluate(val_targets, val_probs, threshold=tau_star)
            test_metrics = evaluator.evaluate(test_targets, test_probs, threshold=tau_star)

            results["configurations"][f_cfg] = {
                "feature_count": in_dim,
                "training_duration_seconds": duration,
                "best_epoch": best_epoch,
                "calibrated_threshold": tau_star,
                "validation": val_metrics.to_dict(),
                "test": test_metrics.to_dict(),
            }

            if val_metrics.pr_auc > best_val_prauc:
                best_val_prauc = val_metrics.pr_auc
                selected_feature_config = f_cfg

            logger.info(
                f"{f_cfg.upper()} (dim={in_dim}): "
                f"Val PR-AUC={val_metrics.pr_auc:.4f}, Val F1={val_metrics.f1_illicit:.4f} | "
                f"Test F1={test_metrics.f1_illicit:.4f}, Test PR-AUC={test_metrics.pr_auc:.4f}, "
                f"Precision={test_metrics.precision_illicit:.4f}, Recall={test_metrics.recall_illicit:.4f}"
            )

        results["selected_feature_config_by_validation"] = selected_feature_config
        logger.info(f"\nFeature configuration selected based strictly on validation: '{selected_feature_config}' (Val PR-AUC: {best_val_prauc:.4f})")

        json_path = self.results_dir / "exp03_feature_ablation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved EXP-03 results to {json_path}")

        return results

    def run_exp04_loss_comparison(
        self,
        best_architecture: str = "graphsage",
        feature_config: str = "original_all",
    ) -> Dict:
        """
        Executes EXP-04: Class Imbalance Loss Comparison on best GNN architecture.
        Compares: cross_entropy, weighted_cross_entropy, focal_loss.
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"EXECUTING EXP-04: LOSS FUNCTION COMPARISON (Architecture: {best_architecture.upper()})")
        logger.info("=" * 80)

        in_dim = self.loader.get_feature_matrix(self.graphs[1], config=feature_config).size(1)
        loss_types = ["cross_entropy", "weighted_cross_entropy", "focal"]

        results = {
            "experiment": "EXP-04",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "architecture": best_architecture,
            "feature_configuration": feature_config,
            "feature_count": in_dim,
            "imbalance_weights": {
                "licit_weight": 1.0,
                "illicit_weight": round(self.scale_pos_weight, 2),
            },
            "loss_functions": {},
        }

        evaluator = ModelEvaluator()
        best_val_prauc = -1.0
        selected_loss = "focal"

        for l_type in loss_types:
            logger.info(f"\n--- Evaluating Loss Function: {l_type.upper()} ---")

            if best_architecture == "graphsage":
                model = GraphSAGENet(in_features=in_dim, hidden_dim=128, dropout=0.3, aggr="mean")
            elif best_architecture == "gcn":
                model = GCNNet(in_features=in_dim, hidden_dim=128, dropout=0.3)
            elif best_architecture == "gat":
                model = GATNet(in_features=in_dim, hidden_dim=32, heads=4, dropout=0.3)
            else:
                raise ValueError(f"Unknown architecture: {best_architecture}")

            t0 = time.time()
            trained_model, history, best_epoch = self.train_model(
                model=model,
                feature_config=feature_config,
                loss_type=l_type,
                max_epochs=40,
                patience=15,
            )
            duration = round(time.time() - t0, 2)

            val_probs, val_targets = self.predict_timesteps(trained_model, range(35, 40), feature_config)
            test_probs, test_targets = self.predict_timesteps(trained_model, range(40, 50), feature_config)

            tau_star, val_best_f1 = evaluator.calibrate_threshold(val_targets, val_probs)
            val_metrics = evaluator.evaluate(val_targets, val_probs, threshold=tau_star)
            test_metrics = evaluator.evaluate(test_targets, test_probs, threshold=tau_star)

            results["loss_functions"][l_type] = {
                "training_duration_seconds": duration,
                "best_epoch": best_epoch,
                "calibrated_threshold": tau_star,
                "validation": val_metrics.to_dict(),
                "test": test_metrics.to_dict(),
            }

            if val_metrics.pr_auc > best_val_prauc:
                best_val_prauc = val_metrics.pr_auc
                selected_loss = l_type

            logger.info(
                f"{l_type.upper()}: "
                f"Val PR-AUC={val_metrics.pr_auc:.4f}, Val F1={val_metrics.f1_illicit:.4f} | "
                f"Test F1={test_metrics.f1_illicit:.4f}, Test PR-AUC={test_metrics.pr_auc:.4f}, "
                f"Precision={test_metrics.precision_illicit:.4f}, Recall={test_metrics.recall_illicit:.4f}"
            )

        results["selected_loss_by_validation"] = selected_loss
        logger.info(f"\nLoss function selected based strictly on validation: '{selected_loss}' (Val PR-AUC: {best_val_prauc:.4f})")

        json_path = self.results_dir / "exp04_loss_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved EXP-04 results to {json_path}")

        return results
