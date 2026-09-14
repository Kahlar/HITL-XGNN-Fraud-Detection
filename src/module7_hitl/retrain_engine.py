"""Fine-tuning and retraining engine incorporating verified feedback buffers into GraphSAGE."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.optim as optim

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader, get_graph_tx_ids
from src.module4_gnn.loss import FocalLoss
from src.module4_gnn.models.graphsage import GraphSAGENet
from src.module7_hitl.feedback_buffer import FeedbackBuffer

logger = get_logger("Module7.RetrainEngine")


@dataclass
class RetrainedModelMetadata:
    """Metadata schema tracking model lineage, feedback provenance, and retraining parameters."""
    model_version: str
    base_model_checkpoint: str
    architecture: str
    feature_configuration: str
    feature_count: int
    loss_function: str
    feedback_strategy: str
    feedback_budget_ratio: float
    feedback_sample_count: int
    feedback_timesteps: List[int]
    training_timesteps: List[int]
    test_timesteps: List[int]
    retraining_epochs: int
    learning_rate: float
    decision_threshold: float
    test_f1: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    test_roc_auc: float
    test_accuracy: float
    saved_checkpoint_path: str
    retrained_at_utc: str

    def to_dict(self) -> Dict:
        return asdict(self)


class RetrainEngine:
    """Fine-tunes GraphSAGE models with verified feedback buffers while guaranteeing zero test leakage."""

    def __init__(
        self,
        graphs_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.graphs_dir = Path(graphs_dir) if graphs_dir else default_config.paths.processed_data_dir / "graphs"
        self.models_dir = Path(models_dir) if models_dir else Path("models/gnn/hitl")
        self.loader = TimestepGraphLoader(self.graphs_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def retrain_with_feedback(
        self,
        base_checkpoint_path: Union[str, Path],
        feedback_buffer: FeedbackBuffer,
        strategy_name: str,
        budget_ratio: float,
        epochs: int = 25,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        decision_threshold: float = 0.5517,
        random_seed: int = 42,
    ) -> Tuple[GraphSAGENet, RetrainedModelMetadata, Dict]:
        """
        Fine-tunes base GraphSAGE on Training graphs (t=1..34) + Feedback Buffer items from Validation graphs (t=35..39).

        Guarantees:
        - Only items explicitly inside feedback_buffer are unmasked as training labels in t=35..39.
        - Test partition (t=40..49) is strictly excluded from retraining.
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Load fresh copy of base model
        base_ckpt = Path(base_checkpoint_path)
        logger.info(f"Loading base GraphSAGE from {base_ckpt} for retraining...")
        model = GraphSAGENet.load_checkpoint(base_ckpt, device=str(self.device))
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

        # Pre-index feedback items by timestep and tx_id
        feedback_tx_set = feedback_buffer.get_tx_ids()
        feedback_records_by_ts: Dict[int, Dict[str, int]] = {}
        for r in feedback_buffer.records:
            if r.verdict in ("ILLICIT", "LICIT"):
                ts_dict = feedback_records_by_ts.setdefault(r.timestep, {})
                ts_dict[r.tx_id] = 1 if r.verdict == "ILLICIT" else 0

        # Fine-tuning loop across t=1..34 and active feedback in t=35..39
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            num_graphs = 0

            # 1. Base training graphs t=1..34
            for ts in range(1, 35):
                data = self.loader.load_graph(ts)
                x = self.loader.get_feature_matrix(data, config="original_all").to(self.device)
                edge_index = data.edge_index.to(self.device)
                y = data.y.to(self.device)

                mask = (y != -1)
                if not mask.any():
                    continue

                optimizer.zero_grad()
                logits = model(x, edge_index)
                loss = criterion(logits[mask], y[mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_graphs += 1

            # 2. Feedback graphs t=35..39 (Mask strictly limited to verified feedback items)
            for ts in range(35, 40):
                if ts not in feedback_records_by_ts:
                    continue

                data = self.loader.load_graph(ts)
                x = self.loader.get_feature_matrix(data, config="original_all").to(self.device)
                edge_index = data.edge_index.to(self.device)
                node_tx_ids = get_graph_tx_ids(data)

                # Construct mask strictly for feedback items
                ts_feedback = feedback_records_by_ts[ts]
                feedback_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
                feedback_y = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)

                for idx, tx_id in enumerate(node_tx_ids):
                    if tx_id in ts_feedback:
                        feedback_mask[idx] = True
                        feedback_y[idx] = ts_feedback[tx_id]

                if not feedback_mask.any():
                    continue

                optimizer.zero_grad()
                logits = model(x, edge_index)
                loss = criterion(logits[feedback_mask], feedback_y[feedback_mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_graphs += 1

        # -------------------------------------------------------------
        # Evaluate on untouched test partition (t=40..49)
        # -------------------------------------------------------------
        model.eval()
        test_y_true: List[int] = []
        test_probs: List[float] = []

        for ts in range(40, 50):
            data = self.loader.load_graph(ts)
            x = self.loader.get_feature_matrix(data, config="original_all").to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.numpy()

            mask = (y != -1)
            with torch.no_grad():
                logits = model(x, edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            test_y_true.extend(y[mask])
            test_probs.extend(probs[mask])

        y_true_arr = np.array(test_y_true, dtype=int)
        probs_arr = np.array(test_probs, dtype=float)
        preds_arr = (probs_arr >= decision_threshold).astype(int)

        prec_arr, rec_arr, _ = precision_recall_curve(y_true_arr, probs_arr, pos_label=1)
        pr_auc_val = float(auc(rec_arr, prec_arr))
        roc_auc_val = float(roc_auc_score(y_true_arr, probs_arr))
        f1_val = float(f1_score(y_true_arr, preds_arr, pos_label=1, zero_division=0))
        prec_val = float(precision_score(y_true_arr, preds_arr, pos_label=1, zero_division=0))
        rec_val = float(recall_score(y_true_arr, preds_arr, pos_label=1, zero_division=0))
        acc_val = float(accuracy_score(y_true_arr, preds_arr))

        # Save retrained model checkpoint
        version_tag = f"graphsage_hitl_{strategy_name}_{int(budget_ratio*100):02d}"
        ckpt_save_path = self.models_dir / f"{version_tag}.pt"
        model.save_checkpoint(ckpt_save_path)

        metadata = RetrainedModelMetadata(
            model_version=version_tag,
            base_model_checkpoint=str(base_ckpt),
            architecture="GraphSAGE",
            feature_configuration="original_all",
            feature_count=165,
            loss_function="focal",
            feedback_strategy=strategy_name,
            feedback_budget_ratio=budget_ratio,
            feedback_sample_count=len(feedback_buffer),
            feedback_timesteps=[35, 39],
            training_timesteps=[1, 34],
            test_timesteps=[40, 49],
            retraining_epochs=epochs,
            learning_rate=lr,
            decision_threshold=decision_threshold,
            test_f1=round(f1_val, 4),
            test_pr_auc=round(pr_auc_val, 4),
            test_precision=round(prec_val, 4),
            test_recall=round(rec_val, 4),
            test_roc_auc=round(roc_auc_val, 4),
            test_accuracy=round(acc_val, 4),
            saved_checkpoint_path=str(ckpt_save_path),
            retrained_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        eval_dict = {
            "y_true": y_true_arr.tolist(),
            "probs": probs_arr.tolist(),
        }

        return model, metadata, eval_dict
