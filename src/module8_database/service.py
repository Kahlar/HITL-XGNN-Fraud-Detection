"""High-level database service orchestrator for migrations, seeding, and transactions."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.common.logger import get_logger
from src.module8_database.models.base import Base
from src.module8_database.models.feedback import AnalystFeedback
from src.module8_database.models.model_registry import ModelRegistry
from src.module8_database.models.transaction import Transaction, TransactionEdge
from src.module8_database.repositories.feedback_repo import FeedbackRepository
from src.module8_database.repositories.model_repo import ModelRegistryRepository
from src.module8_database.repositories.transaction_repo import TransactionRepository

logger = get_logger("Module8.Service")


class DatabaseService:
    """Orchestrates database migrations, artifact ingestion, and transaction operations."""

    def __init__(self):
        self.tx_repo = TransactionRepository()
        self.feedback_repo = FeedbackRepository()
        self.model_repo = ModelRegistryRepository()

    @staticmethod
    async def create_all_tables(engine: AsyncEngine) -> None:
        """Initializes database schema by executing create_all on the metadata."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema tables created successfully.")

    @staticmethod
    async def drop_all_tables(engine: AsyncEngine) -> None:
        """Drops all database tables."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("All database tables dropped.")

    async def seed_model_registry_from_exp07(
        self,
        session: AsyncSession,
        exp07_json_path: Path,
    ) -> int:
        """
        Populates the model_registry table with the retrained active learning model metadata.
        """
        if not exp07_json_path.exists():
            logger.warning(f"EXP-07 JSON file not found at {exp07_json_path}. Skipping model seeding.")
            return 0

        with open(exp07_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta_list = data.get("retrained_models_metadata", [])
        best_tag = data.get("best_active_configuration", {}).get("model_version", "")

        seeded_count = 0
        for m in meta_list:
            model_entry = ModelRegistry(
                model_version=m["model_version"],
                base_model_version=m.get("base_model_checkpoint", "graphsage_best"),
                architecture=m.get("architecture", "GraphSAGE"),
                feature_set=m.get("feature_configuration", "original_all"),
                feature_count=m.get("feature_count", 165),
                loss_function=m.get("loss_function", "focal"),
                decision_threshold=m.get("decision_threshold", 0.5517),
                feedback_strategy=m.get("feedback_strategy"),
                feedback_budget_ratio=m.get("feedback_budget_ratio"),
                feedback_sample_count=m.get("feedback_sample_count"),
                test_f1=m.get("test_f1"),
                test_pr_auc=m.get("test_pr_auc"),
                test_precision=m.get("test_precision"),
                test_recall=m.get("test_recall"),
                test_roc_auc=m.get("test_roc_auc"),
                test_accuracy=m.get("test_accuracy"),
                checkpoint_path=m.get("saved_checkpoint_path", ""),
                is_active_for_inference=(m["model_version"] == best_tag),
            )
            await self.model_repo.register_model(session, model_entry)
            seeded_count += 1

        logger.info(f"Seeded {seeded_count} models into model_registry from {exp07_json_path}")
        return seeded_count
