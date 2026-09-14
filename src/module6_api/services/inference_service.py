"""Inference service for active GNN model management and live transaction scoring."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from src.common.config import default_config
from src.common.logger import get_logger
from src.module4_gnn.models.graphsage import GraphSAGENet
from src.module7_hitl.triage import TriageItem, TriageRouter

logger = get_logger("Module6.InferenceService")


class InferenceService:
    """Manages active GNN model in-memory and provides live transaction scoring."""

    def __init__(self, model_checkpoint_path: Optional[Path] = None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.decision_threshold = 0.5517
        self.triage_router = TriageRouter(decision_threshold=self.decision_threshold)
        self.model: Optional[GraphSAGENet] = None
        self.model_version: str = "graphsage_hitl_uncertainty_20"

        # Determine checkpoint path
        if model_checkpoint_path and model_checkpoint_path.exists():
            self.checkpoint_path = model_checkpoint_path
        else:
            hitl_ckpt = Path("models/gnn/hitl/graphsage_hitl_uncertainty_20.pt")
            base_ckpt = Path("models/gnn/graphsage_best.pt")
            if hitl_ckpt.exists():
                self.checkpoint_path = hitl_ckpt
                self.model_version = "graphsage_hitl_uncertainty_20"
            elif base_ckpt.exists():
                self.checkpoint_path = base_ckpt
                self.model_version = "graphsage_base"
            else:
                self.checkpoint_path = None
                self.model_version = "none"

        self._load_model()

    def _load_model(self) -> None:
        """Loads the active model into memory."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.model = GraphSAGENet.load_checkpoint(self.checkpoint_path, device=str(self.device))
                self.model.eval()
                logger.info(f"Loaded active GNN model ({self.model_version}) from {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load GNN model from {self.checkpoint_path}: {e}")
                self.model = None
        else:
            logger.warning("No GNN model checkpoint found on filesystem.")
            self.model = None

    @property
    def is_available(self) -> bool:
        """Checks if the GNN model is loaded and ready for inference."""
        return self.model is not None

    def score_transaction(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        local_idx: int,
        tx_id: str,
        timestep: int,
    ) -> TriageItem:
        """Runs model inference on a local computational subgraph or timestep graph."""
        if not self.is_available:
            raise RuntimeError("GNN model is not available for inference.")

        # Squeeze feature matrix to 165 if required
        if x.shape[1] > 165:
            x = x[:, :165]

        x_dev = x.to(self.device)
        edge_dev = edge_index.to(self.device)

        with torch.no_grad():
            logits = self.model(x_dev, edge_dev)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            prob = float(probs[local_idx])

        return self.triage_router.triage_candidate(
            tx_id=tx_id,
            timestep=timestep,
            global_node_index=local_idx,
            predicted_prob=prob,
        )

    def score_timestep_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Batch predicts fraud probabilities for all nodes in a timestep graph."""
        if not self.is_available:
            raise RuntimeError("GNN model is not available for inference.")

        if x.shape[1] > 165:
            x = x[:, :165]

        x_dev = x.to(self.device)
        edge_dev = edge_index.to(self.device)

        with torch.no_grad():
            logits = self.model(x_dev, edge_dev)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        return probs
