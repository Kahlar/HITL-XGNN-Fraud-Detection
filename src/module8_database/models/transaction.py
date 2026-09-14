"""ORM models for transactions and directed payment graph edges."""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.module8_database.models.base import Base, TimestampMixin, utc_now


class Transaction(Base, TimestampMixin):
    """Bitcoin transaction entity with model risk assessment and triage state."""
    __tablename__ = "transactions"

    tx_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    timestep: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    ground_truth_label: Mapped[int] = mapped_column(Integer, nullable=False, default=-1) # -1: unknown, 0: licit, 1: illicit

    # Inference predictions
    predicted_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Risk & Triage Priority
    risk_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    uncertainty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    priority_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)

    # Workflow state
    is_triaged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    triage_status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False) # 'PENDING', 'QUEUED', 'REVIEWED', 'ESCALATED'

    # Relationships
    explanations: Mapped[List["TransactionExplanation"]] = relationship(
        "TransactionExplanation",
        back_populates="transaction",
        cascade="all, delete-orphan",
    )
    feedbacks: Mapped[List["AnalystFeedback"]] = relationship(
        "AnalystFeedback",
        back_populates="transaction",
        cascade="all, delete-orphan",
    )
    outgoing_edges: Mapped[List["TransactionEdge"]] = relationship(
        "TransactionEdge",
        foreign_keys="TransactionEdge.source_tx_id",
        back_populates="source_transaction",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[List["TransactionEdge"]] = relationship(
        "TransactionEdge",
        foreign_keys="TransactionEdge.target_tx_id",
        back_populates="target_transaction",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_transactions_timestep_triaged", "timestep", "is_triaged"),
        Index("ix_transactions_priority_risk", "priority_score", "risk_level"),
    )


class TransactionEdge(Base):
    """Directed transaction-to-transaction payment flow edge."""
    __tablename__ = "transaction_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_tx_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_tx_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestep: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    source_transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        foreign_keys=[source_tx_id],
        back_populates="outgoing_edges",
    )
    target_transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        foreign_keys=[target_tx_id],
        back_populates="incoming_edges",
    )

    __table_args__ = (
        Index("ix_edges_source_target_ts", "source_tx_id", "target_tx_id", "timestep"),
    )
