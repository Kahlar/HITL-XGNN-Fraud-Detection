"""ORM model for GNNExplainer transaction explanation records."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.module8_database.models.base import Base, utc_now


class TransactionExplanation(Base):
    """GNNExplainer post-hoc attribution artifact for a target transaction."""
    __tablename__ = "transaction_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Faithfulness & Sparsity Metrics
    fidelity_plus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fidelity_minus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_sparsity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feature_sparsity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generation_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Attribution JSON Payloads
    edge_attributions: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    feature_attributions: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    subgraph_nodes: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    subgraph_edges: Mapped[Optional[List[List[Any]]]] = mapped_column(JSON, nullable=True)
    feature_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        back_populates="explanations",
    )

    __table_args__ = (
        Index("ix_explanations_tx_model", "tx_id", "model_version"),
    )
