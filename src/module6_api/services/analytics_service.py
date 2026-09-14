"""Analytics aggregation service integrating offline benchmarks and live database metrics."""

import json
from pathlib import Path
from typing import Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import default_config
from src.common.logger import get_logger
from src.module6_api.schemas.analytics import (
    AnalyticsMetricsResponse,
    ModelPerformanceSummary,
    TemporalMetricItem,
)
from src.module8_database.repositories.feedback_repo import FeedbackRepository
from src.module8_database.repositories.model_repo import ModelRegistryRepository
from src.module8_database.repositories.transaction_repo import TransactionRepository

logger = get_logger("Module6.AnalyticsService")


class AnalyticsService:
    """Consolidates benchmark artifacts and real-time database state for dashboard telemetry."""

    def __init__(self):
        self.exp_dir = default_config.paths.processed_data_dir / "experiments"
        self.tx_repo = TransactionRepository()
        self.feedback_repo = FeedbackRepository()
        self.model_repo = ModelRegistryRepository()

    async def get_dashboard_metrics(self, session: AsyncSession) -> AnalyticsMetricsResponse:
        """Constructs master analytics payload."""
        # 1. Dataset Summary
        dataset_summary = {
            "total_transactions": 203769,
            "total_edges": 234355,
            "total_timesteps": 49,
            "total_licit": 42019,
            "total_illicit": 4545,
            "total_unlabeled": 157205,
            "imbalance_ratio": 9.25,
        }

        # 2. Active Model Performance
        active_db_model = await self.model_repo.get_active_model(session)
        if active_db_model:
            active_model = ModelPerformanceSummary(
                model_version=active_db_model.model_version,
                architecture=active_db_model.architecture,
                decision_threshold=active_db_model.decision_threshold,
                test_f1=active_db_model.test_f1 or 0.5030,
                test_pr_auc=active_db_model.test_pr_auc or 0.4358,
                test_precision=active_db_model.test_precision or 0.7086,
                test_recall=active_db_model.test_recall or 0.3899,
                test_roc_auc=active_db_model.test_roc_auc or 0.8250,
            )
        else:
            active_model = ModelPerformanceSummary(
                model_version="graphsage_hitl_uncertainty_20",
                architecture="GraphSAGE",
                decision_threshold=0.5517,
                test_f1=0.5030,
                test_pr_auc=0.4358,
                test_precision=0.7086,
                test_recall=0.3899,
                test_roc_auc=0.8250,
            )

        # 3. Temporal Drift Benchmarks from EXP-06
        temporal_items: List[TemporalMetricItem] = []
        exp06_file = self.exp_dir / "exp06_temporal_drift.json"
        if exp06_file.exists():
            try:
                with open(exp06_file, "r", encoding="utf-8") as f:
                    drift_data = json.load(f)
                raw_metrics = drift_data.get("per_timestep_metrics")
                if raw_metrics is None:
                    raw_metrics = drift_data.get("timestep_evaluations", [])

                for ts_info in raw_metrics:
                    num_labeled = ts_info.get("labeled_nodes", ts_info.get("num_labeled", 0))
                    num_illicit = ts_info.get("illicit_count", ts_info.get("num_illicit", 0))

                    if "illicit_prevalence" in ts_info:
                        prevalence_pct = round(float(ts_info["illicit_prevalence"]) * 100.0, 4)
                    else:
                        prevalence_pct = float(ts_info.get("illicit_prevalence_pct", 0.0))

                    f1 = float(ts_info.get("f1_illicit", ts_info.get("f1_score", 0.0)))
                    precision = float(ts_info.get("precision_illicit", ts_info.get("precision", 0.0)))
                    recall = float(ts_info.get("recall_illicit", ts_info.get("recall", 0.0)))
                    pr_auc = float(ts_info.get("pr_auc", 0.0))

                    temporal_items.append(
                        TemporalMetricItem(
                            timestep=ts_info["timestep"],
                            period=ts_info.get("period", "unknown"),
                            num_labeled=num_labeled,
                            num_illicit=num_illicit,
                            illicit_prevalence_pct=prevalence_pct,
                            f1_score=f1,
                            pr_auc=pr_auc,
                            precision=precision,
                            recall=recall,
                        )
                    )
            except Exception as e:
                logger.error(f"Error loading EXP-06 metrics: {e}")

        # 4. Active Learning Benchmarks from EXP-07
        al_benchmarks = {}
        exp07_file = self.exp_dir / "exp07_hitl_summary.json"
        if exp07_file.exists():
            try:
                with open(exp07_file, "r", encoding="utf-8") as f:
                    al_benchmarks = json.load(f)
            except Exception as e:
                logger.error(f"Error loading EXP-07 metrics: {e}")

        # 5. Live Feedback Stats from Database
        feedback_stats = await self.feedback_repo.get_feedback_statistics(session)

        return AnalyticsMetricsResponse(
            dataset_summary=dataset_summary,
            active_model=active_model,
            temporal_drift=temporal_items,
            active_learning_benchmarks=al_benchmarks,
            feedback_stats=feedback_stats,
        )
