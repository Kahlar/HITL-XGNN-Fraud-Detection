"""Graph service for loading timestep graphs and extracting frontend-ready 2-hop subgraphs."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.pyg_builder import TimestepGraphLoader
from src.module3_graph_builder.subgraph_extractor import SubgraphExtractor
from src.module6_api.schemas.graph import GraphEdge, GraphNode, SubgraphResponse

logger = get_logger("Module6.GraphService")


class GraphService:
    """Provides fast local subgraph extraction and serialization for web visualizations."""

    def __init__(self, graphs_dir: Optional[Path] = None):
        self.graphs_dir = graphs_dir or default_config.paths.processed_data_dir / "graphs"
        self.loader = TimestepGraphLoader(self.graphs_dir)
        self.extractor = SubgraphExtractor(k_default=2)

    def get_subgraph(
        self,
        tx_id: str,
        timestep: int,
        k_hops: int = 2,
        predictions_map: Optional[Dict[str, float]] = None,
        edge_weights_map: Optional[Dict[tuple, float]] = None,
    ) -> SubgraphResponse:
        """
        Extracts k-hop computational neighborhood around tx_id and formats as GraphNode and GraphEdge objects.
        """
        graph_data = self.loader.load_graph(timestep)

        # Extract induced neighborhood
        subgraph_data = self.extractor.extract_k_hop_subgraph(
            graph_data, target=tx_id, k=k_hops, flow="both"
        )

        data = subgraph_data.subgraph
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        # Calculate degrees
        adj = data.edge_index.cpu().numpy()
        out_degrees = np.bincount(adj[0], minlength=num_nodes) if num_nodes > 0 else []
        in_degrees = np.bincount(adj[1], minlength=num_nodes) if num_nodes > 0 else []

        nodes: List[GraphNode] = []
        for loc_idx in range(num_nodes):
            curr_tx = subgraph_data.node_tx_ids[loc_idx]
            y_val = int(data.y[loc_idx].item()) if hasattr(data, "y") and data.y is not None else -1
            prob = predictions_map.get(curr_tx) if predictions_map else None

            risk = None
            if prob is not None:
                if prob >= 0.85:
                    risk = "CRITICAL"
                elif prob >= 0.70:
                    risk = "HIGH"
                elif prob >= 0.40:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

            nodes.append(
                GraphNode(
                    id=curr_tx,
                    label=f"{curr_tx[:8]}...",
                    timestep=timestep,
                    predicted_prob=round(prob, 4) if prob is not None else None,
                    predicted_class=int(prob >= 0.5517) if prob is not None else None,
                    risk_level=risk,
                    ground_truth=y_val,
                    is_target=(curr_tx == tx_id),
                    in_degree=int(in_degrees[loc_idx]) if len(in_degrees) > 0 else 0,
                    out_degree=int(out_degrees[loc_idx]) if len(out_degrees) > 0 else 0,
                )
            )

        edges: List[GraphEdge] = []
        for e_idx in range(num_edges):
            src_loc = int(adj[0, e_idx])
            dst_loc = int(adj[1, e_idx])
            src_tx = subgraph_data.node_tx_ids[src_loc]
            dst_tx = subgraph_data.node_tx_ids[dst_loc]

            weight = None
            if edge_weights_map:
                weight = edge_weights_map.get((src_tx, dst_tx))

            edges.append(
                GraphEdge(
                    id=f"{src_tx}->{dst_tx}",
                    source=src_tx,
                    target=dst_tx,
                    timestep=timestep,
                    importance_weight=weight,
                )
            )

        # Check if pre-computed explanation exists
        expl_file = default_config.paths.processed_data_dir / "experiments" / "exp05_explanations" / f"tx_{tx_id}.json"
        explanation_available = expl_file.exists()

        return SubgraphResponse(
            target_tx_id=tx_id,
            timestep=timestep,
            k_hops=k_hops,
            num_nodes=num_nodes,
            num_edges=num_edges,
            nodes=nodes,
            edges=edges,
            explanation_available=explanation_available,
        )
