"""Pydantic schemas for frontend graph canvas and 2-hop subgraphs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GraphNode(BaseModel):
    """Represents a transaction node in the visualization canvas."""
    id: str                    # tx_id string
    label: str                 # Human-readable short label
    timestep: int
    predicted_prob: Optional[float] = None
    predicted_class: Optional[int] = None
    risk_level: Optional[str] = None
    ground_truth: int = -1
    is_target: bool = False
    in_degree: int = 0
    out_degree: int = 0


class GraphEdge(BaseModel):
    """Represents a directed payment link between two transactions."""
    id: str                    # e.g. "src->dst"
    source: str                # source tx_id
    target: str                # target tx_id
    timestep: int
    importance_weight: Optional[float] = None
    rank: Optional[int] = None


class SubgraphResponse(BaseModel):
    """Frontend-ready 2-hop induced computational subgraph response."""
    target_tx_id: str
    timestep: int
    k_hops: int
    num_nodes: int
    num_edges: int
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    explanation_available: bool = False
