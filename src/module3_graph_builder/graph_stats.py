"""Topological and statistical metrics calculator for PyG temporal graphs."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.common.config import default_config
from src.common.logger import get_logger

logger = get_logger("Module3.GraphStats")


class GraphStatisticsCalculator:
    """Computes comprehensive topological and demographic metrics per timestep graph."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else default_config.paths.processed_data_dir

    def calculate_timestep_statistics(self, data: Data) -> Dict[str, Union[int, float, Dict]]:
        """Calculates topological and class metrics for a single timestep graph."""
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0

        # Degree calculations
        if num_edges > 0:
            src = data.edge_index[0]
            dst = data.edge_index[1]
            out_degrees = torch.bincount(src, minlength=num_nodes).numpy()
            in_degrees = torch.bincount(dst, minlength=num_nodes).numpy()
            total_degrees = in_degrees + out_degrees

            avg_in = float(np.mean(in_degrees))
            avg_out = float(np.mean(out_degrees))
            max_in = int(np.max(in_degrees))
            max_out = int(np.max(out_degrees))
            max_total = int(np.max(total_degrees))
            num_isolated = int(np.sum(total_degrees == 0))
        else:
            avg_in = 0.0
            avg_out = 0.0
            max_in = 0
            max_out = 0
            max_total = 0
            num_isolated = num_nodes

        # Graph Density: E / (V * (V - 1)) for directed graph
        if num_nodes > 1:
            density = float(num_edges / (num_nodes * (num_nodes - 1)))
        else:
            density = 0.0

        # Label counts
        y_np = data.y.numpy()
        illicit_cnt = int(np.sum(y_np == 1))
        licit_cnt = int(np.sum(y_np == 0))
        unknown_cnt = int(np.sum(y_np == -1))
        labeled_cnt = illicit_cnt + licit_cnt

        pct_labeled = round((labeled_cnt / num_nodes) * 100, 2) if num_nodes > 0 else 0.0
        pct_illicit_in_labeled = round((illicit_cnt / labeled_cnt) * 100, 2) if labeled_cnt > 0 else 0.0
        imbalance_ratio = round(licit_cnt / illicit_cnt, 2) if illicit_cnt > 0 else 0.0

        # Connected Components using NetworkX
        if num_edges > 0:
            # Build directed graph
            edge_list = list(zip(src.tolist(), dst.tolist()))
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edge_list)
            num_weakly_connected = int(nx.number_weakly_connected_components(G))
            num_strongly_connected = int(nx.number_strongly_connected_components(G))
        else:
            num_weakly_connected = num_nodes
            num_strongly_connected = num_nodes

        return {
            "time_step": int(data.time_step),
            "split": str(data.split) if hasattr(data, "split") else "unknown",
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "labeled_nodes": labeled_cnt,
            "illicit_count": illicit_cnt,
            "licit_count": licit_cnt,
            "unknown_count": unknown_cnt,
            "pct_labeled": pct_labeled,
            "pct_illicit_in_labeled": pct_illicit_in_labeled,
            "imbalance_ratio": imbalance_ratio,
            "avg_in_degree": round(avg_in, 4),
            "avg_out_degree": round(avg_out, 4),
            "max_in_degree": max_in,
            "max_out_degree": max_out,
            "max_total_degree": max_total,
            "num_isolated_nodes": num_isolated,
            "graph_density": round(density, 8),
            "weakly_connected_components": num_weakly_connected,
            "strongly_connected_components": num_strongly_connected,
        }

    def calculate_all_and_save(self, graphs: Dict[int, Data]) -> Dict:
        """Computes statistics for all 49 timesteps and saves JSON + TXT reports."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Computing graph statistics for {len(graphs)} timestep graphs...")

        per_ts_stats: Dict[int, Dict] = {}
        total_nodes = 0
        total_edges = 0
        total_illicit = 0
        total_licit = 0
        total_unknown = 0
        total_isolated = 0

        for ts in sorted(graphs.keys()):
            stats = self.calculate_timestep_statistics(graphs[ts])
            per_ts_stats[ts] = stats
            
            total_nodes += stats["num_nodes"]
            total_edges += stats["num_edges"]
            total_illicit += stats["illicit_count"]
            total_licit += stats["licit_count"]
            total_unknown += stats["unknown_count"]
            total_isolated += stats["num_isolated_nodes"]

        overall_stats = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_timesteps": len(graphs),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_illicit": total_illicit,
            "total_licit": total_licit,
            "total_unknown": total_unknown,
            "total_labeled": total_illicit + total_licit,
            "total_isolated_nodes": total_isolated,
            "overall_pct_illicit_in_labeled": round((total_illicit / (total_illicit + total_licit)) * 100, 2) if (total_illicit + total_licit) > 0 else 0.0,
            "overall_imbalance_ratio": round(total_licit / total_illicit, 2) if total_illicit > 0 else 0.0,
            "per_timestep_statistics": per_ts_stats,
        }

        # Save JSON
        json_path = self.output_dir / "graph_statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(overall_stats, f, indent=2)
        logger.info(f"Saved graph statistics JSON to {json_path}")

        # Save TXT
        txt_path = self.output_dir / "graph_statistics.txt"
        self._write_human_readable_stats(overall_stats, txt_path)
        logger.info(f"Saved graph statistics TXT to {txt_path}")

        return overall_stats

    def _write_human_readable_stats(self, stats: Dict, txt_path: Path) -> None:
        """Formats and writes a comprehensive human-readable report."""
        lines = [
            "=" * 105,
            "ELLIPTIC BITCOIN DATASET — PYTORCH GEOMETRIC TEMPORAL GRAPH STATISTICS",
            "=" * 105,
            f"Generated (UTC)           : {stats['generated_utc']}",
            f"Total Graphs (Timesteps)  : {stats['total_timesteps']}",
            f"Total Nodes Across Graphs : {stats['total_nodes']:,}",
            f"Total Edges Across Graphs : {stats['total_edges']:,}",
            f"Total Illicit (1)         : {stats['total_illicit']:,}",
            f"Total Licit (2)           : {stats['total_licit']:,}",
            f"Total Unknown (-1)        : {stats['total_unknown']:,}",
            f"Total Isolated Nodes      : {stats['total_isolated_nodes']:,}",
            f"Overall Labeled Imbalance : {stats['overall_imbalance_ratio']:.2f} : 1",
            "",
            "-" * 105,
            f"{'TS':<4} | {'Split':<5} | {'Nodes':<6} | {'Edges':<6} | {'Illicit':<7} | {'Licit':<6} | {'Unknown':<7} | {'Avg In/Out':<10} | {'Max Deg':<7} | {'Isolated':<8} | {'Weakly CC':<9}",
            "-" * 105,
        ]

        for ts, s in sorted(stats["per_timestep_statistics"].items(), key=lambda x: int(x[0])):
            lines.append(
                f"{ts:<4} | {s['split']:<5} | {s['num_nodes']:>6,d} | {s['num_edges']:>6,d} | "
                f"{s['illicit_count']:>7,d} | {s['licit_count']:>6,d} | {s['unknown_count']:>7,d} | "
                f"{s['avg_in_degree']:>10.3f} | {s['max_total_degree']:>7d} | {s['num_isolated_nodes']:>8,d} | {s['weakly_connected_components']:>9,d}"
            )

        lines.append("=" * 105)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
