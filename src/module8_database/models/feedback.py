"""ORM model for human analyst and simulated reviewer feedback submissions."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.module8_database.models.base import Base, utc_now


class AnalystFeedback(Base):
    """Auditable review verdict and case investigation feedback."""
    __tablename__ = "analyst_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analyst_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True) # 'ILLICIT', 'LICIT', 'ESCALATED', 'INCONCLUSIVE'
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)            # 1 to 5
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    model_predicted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    explanation_viewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback_source: Mapped[str] = mapped_column(String(32), nullable=False)    # 'SIMULATED_ORACLE', 'HUMAN_ANALYST'

    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction",
        back_populates="feedbacks",
    )

    __table_args__ = (
        Index("ix_feedback_analyst_verdict", "analyst_id", "verdict"),
        Index("ix_feedback_reviewed_at", "reviewed_at"),
    )
