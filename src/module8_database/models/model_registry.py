"""ORM model for model lineage tracking, checkpoints, and performance registry."""

from typing import Optional
from sqlalchemy import Boolean, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.module8_database.models.base import Base, TimestampMixin


class ModelRegistry(Base, TimestampMixin):
    """Tracks deployed and retrained model versions, provenance, and evaluation metrics."""
    __tablename__ = "model_registry"

    model_version: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    base_model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    architecture: Mapped[str] = mapped_column(String(32), nullable=False) # 'GraphSAGE', 'GAT', 'GCN'
    feature_set: Mapped[str] = mapped_column(String(32), default="original_all", nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, default=165, nullable=False)
    loss_function: Mapped[str] = mapped_column(String(32), default="focal", nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, default=0.5517, nullable=False)

    # Active learning lineage
    feedback_strategy: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    feedback_budget_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feedback_sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Evaluation metrics
    test_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_pr_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Storage & Deployment
    checkpoint_path: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active_for_inference: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    __table_args__ = (
        Index("ix_model_registry_active_arch", "is_active_for_inference", "architecture"),
    )
