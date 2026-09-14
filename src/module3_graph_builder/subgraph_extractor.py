"""k-hop computational subgraph extractor for transaction investigation and XAI."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.common.logger import get_logger

logger = get_logger("Module3.SubgraphExtractor")


@dataclass
class SubgraphData:
    """Container holding extracted subgraph Data and bidirectional index mappings."""
    subgraph: Data
    target_tx_id: str
    target_local_idx: int
    target_global_idx: int
    k_hops: int
    num_nodes: int
    num_edges: int
    node_tx_ids: List[str]
    local_to_global_indices: List[int]
    global_to_local_indices: Dict[int, int]


class SubgraphExtractor:
    """Extracts k-hop computational subgraphs around target transactions."""

    def __init__(self, k_default: int = 2):
        self.k_default = k_default

    def extract_k_hop_subgraph(
        self,
        data: Data,
        target: Union[str, int],
        k: Optional[int] = None,
        flow: str = "both"
    ) -> SubgraphData:
        """
        Extracts the k-hop induced neighborhood around a target transaction.

        Args:
            data: PyG Data object for the timestep graph.
            target: Target transaction string txId or integer node index.
            k: Number of hops (defaults to self.k_default, usually 2).
            flow: Directional expansion mode:
                  - 'both': Undirected 2-hop computational neighborhood (standard for XAI & UI).
                  - 'incoming': Only payment flows feeding into the target (ancestors).
                  - 'outgoing': Only payment flows moving out from target (descendants).

        Returns:
            SubgraphData containing induced PyG Data and mapping structures.
        """
        num_hops = k if k is not None else self.k_default

        # Resolve target index and tx_id
        if isinstance(target, str):
            tx_ids = data.tx_id
            if target not in tx_ids:
                raise ValueError(f"Target transaction ID '{target}' not found in provided graph.")
            target_global_idx = tx_ids.index(target)
            target_tx_id = target
        else:
            target_global_idx = int(target)
            if target_global_idx < 0 or target_global_idx >= data.num_nodes:
                raise IndexError(f"Target index {target_global_idx} out of range [0, {data.num_nodes-1}].")
            target_tx_id = data.tx_id[target_global_idx]

        edge_index = data.edge_index

        # If graph has no edges or isolated node
        if edge_index.numel() == 0:
            return self._build_isolated_subgraph(data, target_global_idx, target_tx_id, num_hops)

        # PyG k_hop_subgraph implementation
        # flow parameter: 'source_to_target' or 'target_to_source'
        # For 'both', we pass directed_flow='both' by making edge_index undirected temporarily for traversal
        if flow == "both":
            # Undirected traversal to find all connected 2-hop nodes
            rev_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
            bi_edges = torch.cat([edge_index, rev_edges], dim=1)
            subset_nodes, _, _, _ = k_hop_subgraph(
                node_idx=target_global_idx,
                num_hops=num_hops,
                edge_index=bi_edges,
                relabel_nodes=False,
                num_nodes=data.num_nodes
            )
        elif flow == "incoming":
            # Transactions that send money into target (u -> target)
            subset_nodes, _, _, _ = k_hop_subgraph(
                node_idx=target_global_idx,
                num_hops=num_hops,
                edge_index=edge_index,
                flow="target_to_source",
                relabel_nodes=False,
                num_nodes=data.num_nodes
            )
        elif flow == "outgoing":
            # Transactions target sends money into (target -> v)
            subset_nodes, _, _, _ = k_hop_subgraph(
                node_idx=target_global_idx,
                num_hops=num_hops,
                edge_index=edge_index,
                flow="source_to_target",
                relabel_nodes=False,
                num_nodes=data.num_nodes
            )
        else:
            raise ValueError(f"Unknown flow mode '{flow}'. Expected 'both', 'incoming', or 'outgoing'.")

        # Convert subset nodes to sorted list
        local_to_global = subset_nodes.tolist()
        # Ensure target node is present
        if target_global_idx not in local_to_global:
            local_to_global = [target_global_idx] + local_to_global

        global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(local_to_global)}
        target_local_idx = global_to_local[target_global_idx]
        node_tx_ids = [data.tx_id[g_idx] for g_idx in local_to_global]

        # Extract induced directed edges between subset nodes (preserving original direction!)
        sub_node_set = set(local_to_global)
        u_global = edge_index[0].tolist()
        v_global = edge_index[1].tolist()

        sub_u = []
        sub_v = []
        for u, v in zip(u_global, v_global):
            if u in sub_node_set and v in sub_node_set:
                sub_u.append(global_to_local[u])
                sub_v.append(global_to_local[v])

        if sub_u:
            sub_edge_index = torch.tensor([sub_u, sub_v], dtype=torch.long)
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long)

        # Slice node features and labels
        subset_tensor = torch.tensor(local_to_global, dtype=torch.long)
        sub_x = data.x[subset_tensor]
        sub_x_raw = data.x_raw[subset_tensor] if hasattr(data, "x_raw") and data.x_raw is not None else None
        sub_y = data.y[subset_tensor]

        subgraph_data = Data(
            x=sub_x,
            x_raw=sub_x_raw,
            edge_index=sub_edge_index,
            y=sub_y,
            tx_id=node_tx_ids,
            time_step=data.time_step,
            split=data.split if hasattr(data, "split") else "unknown",
            num_nodes=len(local_to_global),
            target_idx=target_local_idx,
            target_tx_id=target_tx_id,
        )

        return SubgraphData(
            subgraph=subgraph_data,
            target_tx_id=target_tx_id,
            target_local_idx=target_local_idx,
            target_global_idx=target_global_idx,
            k_hops=num_hops,
            num_nodes=len(local_to_global),
            num_edges=sub_edge_index.size(1),
            node_tx_ids=node_tx_ids,
            local_to_global_indices=local_to_global,
            global_to_local_indices=global_to_local,
        )

    def _build_isolated_subgraph(
        self,
        data: Data,
        target_global_idx: int,
        target_tx_id: str,
        k_hops: int
    ) -> SubgraphData:
        """Constructs a 1-node subgraph for an isolated transaction."""
        sub_x = data.x[target_global_idx : target_global_idx + 1]
        sub_x_raw = data.x_raw[target_global_idx : target_global_idx + 1] if hasattr(data, "x_raw") and data.x_raw is not None else None
        sub_y = data.y[target_global_idx : target_global_idx + 1]

        sub_data = Data(
            x=sub_x,
            x_raw=sub_x_raw,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=sub_y,
            tx_id=[target_tx_id],
            time_step=data.time_step,
            split=data.split if hasattr(data, "split") else "unknown",
            num_nodes=1,
            target_idx=0,
            target_tx_id=target_tx_id,
        )

        return SubgraphData(
            subgraph=sub_data,
            target_tx_id=target_tx_id,
            target_local_idx=0,
            target_global_idx=target_global_idx,
            k_hops=k_hops,
            num_nodes=1,
            num_edges=0,
            node_tx_ids=[target_tx_id],
            local_to_global_indices=[target_global_idx],
            global_to_local_indices={target_global_idx: 0},
        )
