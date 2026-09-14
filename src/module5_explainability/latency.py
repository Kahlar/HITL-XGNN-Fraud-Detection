"""Latency profiling and performance benchmarking for explanation inference."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
import numpy as np


@dataclass
class LatencyProfileSummary:
    """Summary of explanation generation runtime across evaluated targets."""
    num_samples: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_duration_seconds: float

    def to_dict(self) -> Dict:
        return asdict(self)


class LatencyProfiler:
    """Collects and summarizes explanation runtime statistics for API readiness."""

    def __init__(self):
        self.latencies_ms: List[float] = []
        self.subgraph_sizes: List[int] = []

    def record(self, latency_ms: float, subgraph_nodes: int = 0) -> None:
        """Records a single explanation run latency."""
        self.latencies_ms.append(float(latency_ms))
        self.subgraph_sizes.append(int(subgraph_nodes))

    def summarize(self) -> LatencyProfileSummary:
        """Computes summary percentiles and distributions."""
        if not self.latencies_ms:
            return LatencyProfileSummary(
                num_samples=0,
                mean_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                total_duration_seconds=0.0,
            )

        arr = np.array(self.latencies_ms)
        return LatencyProfileSummary(
            num_samples=len(arr),
            mean_latency_ms=round(float(np.mean(arr)), 2),
            median_latency_ms=round(float(np.median(arr)), 2),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 2),
            min_latency_ms=round(float(np.min(arr)), 2),
            max_latency_ms=round(float(np.max(arr)), 2),
            total_duration_seconds=round(float(np.sum(arr) / 1000.0), 2),
        )
