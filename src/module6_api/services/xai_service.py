"""XAI service providing GNNExplainer attribution and fidelity caching."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader
from src.module3_graph_builder.subgraph_extractor import SubgraphExtractor
from src.module5_explainability.gnn_explainer import GNNExplainerEngine, TransactionExplanation
from src.module6_api.schemas.explanation import (
    EdgeAttributionItem,
    ExplanationResponse,
    FeatureAttributionItem,
)
from src.module6_api.services.inference_service import InferenceService

logger = get_logger("Module6.XAIService")


class XAIService:
    """Provides fast retrieval of precomputed explanations or on-demand GNNExplainer execution."""

    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.explanations_dir = default_config.paths.processed_data_dir / "experiments" / "exp05_explanations"
        self.loader = TimestepGraphLoader(default_config.paths.processed_data_dir / "graphs")
        self.extractor = SubgraphExtractor(k_default=2)

        # Lazy initialize GNNExplainerEngine
        self._explainer_engine: Optional[GNNExplainerEngine] = None

    @property
    def explainer_engine(self) -> GNNExplainerEngine:
        if self._explainer_engine is None:
            if not self.inference_service.is_available:
                raise RuntimeError("GNN model is not loaded; cannot initialize GNNExplainer.")
            self._explainer_engine = GNNExplainerEngine(
                model=self.inference_service.model,
                epochs=60,
                lr=0.01,
                device=str(self.inference_service.device),
            )
        return self._explainer_engine

    def get_explanation(
        self,
        tx_id: str,
        timestep: int,
    ) -> ExplanationResponse:
        """
        Retrieves explanation from cached artifacts if available, or generates on-demand.
        """
        cached_file = self.explanations_dir / f"tx_{tx_id}.json"
        if cached_file.exists():
            with open(cached_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            top_features = [
                FeatureAttributionItem(
                    feature_index=feat["feature_index"],
                    feature_name=feat["feature_name"],
                    feature_type=feat["feature_type"],
                    raw_importance=feat["raw_importance"],
                    normalized_importance=feat["normalized_importance"],
                    rank=feat["rank"],
                )
                for feat in data.get("top_features", [])
            ]

            top_edges = [
                EdgeAttributionItem(
                    source_tx_id=edge["source_tx_id"],
                    target_tx_id=edge["target_tx_id"],
                    source_global_idx=edge["source_global_idx"],
                    target_global_idx=edge["target_global_idx"],
                    edge_index_in_subgraph=edge["edge_index_in_subgraph"],
                    raw_importance=edge["raw_importance"],
                    normalized_importance=edge["normalized_importance"],
                    rank=edge["rank"],
                )
                for edge in data.get("top_edges", [])
            ]

            prob = data.get("prediction_probability", 0.0)
            risk = "CRITICAL" if prob >= 0.85 else "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"

            return ExplanationResponse(
                tx_id=data["tx_id"],
                timestep=data["timestep"],
                model_version=self.inference_service.model_version,
                prediction_probability=prob,
                predicted_class=data.get("predicted_class", int(prob >= 0.5517)),
                risk_level=risk,
                category=data.get("category", "unlabeled"),
                period=data.get("period", "unknown"),
                subgraph_num_nodes=data.get("subgraph_num_nodes", 0),
                subgraph_num_edges=data.get("subgraph_num_edges", 0),
                fidelity_plus=data.get("fidelity_plus"),
                fidelity_minus=data.get("fidelity_minus"),
                edge_sparsity=data.get("edge_sparsity"),
                feature_sparsity=data.get("feature_sparsity"),
                generation_latency_ms=data.get("generation_latency_ms", 0.0),
                top_features=top_features,
                top_edges=top_edges,
                feature_summary=data.get("feature_summary", {}),
            )

        # On-demand explanation generation
        logger.info(f"Generating on-demand GNNExplainer attribution for transaction {tx_id} (ts={timestep})...")
        graph_data = self.loader.load_graph(timestep)
        if graph_data.x.shape[1] > 165:
            graph_data.x = graph_data.x[:, :165]

        subgraph_data = self.extractor.extract_k_hop_subgraph(graph_data, target=tx_id, k=2, flow="both")
        if subgraph_data.subgraph.x.shape[1] > 165:
            subgraph_data.subgraph.x = subgraph_data.subgraph.x[:, :165]

        expl = self.explainer_engine.explain_subgraph(
            subgraph=subgraph_data,
            target_local_idx=subgraph_data.target_local_idx,
            timestep=timestep,
            threshold=self.inference_service.decision_threshold,
            top_k_features=15,
            top_k_edges=10,
        )

        top_features = [
            FeatureAttributionItem(
                feature_index=feat["feature_index"],
                feature_name=feat["feature_name"],
                feature_type=feat["feature_type"],
                raw_importance=feat["raw_importance"],
                normalized_importance=feat["normalized_importance"],
                rank=feat["rank"],
            )
            for feat in expl.top_features
        ]

        top_edges = [
            EdgeAttributionItem(
                source_tx_id=edge["source_tx_id"],
                target_tx_id=edge["target_tx_id"],
                source_global_idx=edge["source_global_idx"],
                target_global_idx=edge["target_global_idx"],
                edge_index_in_subgraph=edge["edge_index_in_subgraph"],
                raw_importance=edge["raw_importance"],
                normalized_importance=edge["normalized_importance"],
                rank=edge["rank"],
            )
            for edge in expl.top_edges
        ]

        prob = expl.prediction_probability
        risk = "CRITICAL" if prob >= 0.85 else "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"

        return ExplanationResponse(
            tx_id=expl.tx_id,
            timestep=expl.timestep,
            model_version=self.inference_service.model_version,
            prediction_probability=prob,
            predicted_class=expl.predicted_class,
            risk_level=risk,
            category=expl.category,
            period=expl.period,
            subgraph_num_nodes=expl.subgraph_num_nodes,
            subgraph_num_edges=expl.subgraph_num_edges,
            fidelity_plus=expl.fidelity_plus,
            fidelity_minus=expl.fidelity_minus,
            edge_sparsity=expl.edge_sparsity,
            feature_sparsity=expl.feature_sparsity,
            generation_latency_ms=expl.generation_latency_ms,
            top_features=top_features,
            top_edges=top_edges,
            feature_summary=expl.feature_summary,
        )
