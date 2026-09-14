"""Repository for reviewer feedback records and audit logs."""

from typing import Dict, List, Optional, Sequence
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.module8_database.models.feedback import AnalystFeedback
from src.module8_database.models.transaction import Transaction
from src.module8_database.repositories.base_repository import BaseRepository


class FeedbackRepository(BaseRepository[AnalystFeedback]):
    """Data access repository for AnalystFeedback submissions."""

    def __init__(self):
        super().__init__(AnalystFeedback)

    async def submit_feedback(
        self,
        session: AsyncSession,
        feedback: AnalystFeedback,
    ) -> AnalystFeedback:
        """
        Inserts reviewer feedback and updates the parent transaction's workflow status.
        """
        session.add(feedback)
        await session.flush()

        # Update parent transaction status
        tx = await session.get(Transaction, feedback.tx_id)
        if tx:
            tx.triage_status = "ESCALATED" if feedback.verdict == "ESCALATED" else "REVIEWED"
            await session.flush()

        return feedback

    async def get_by_tx(
        self,
        session: AsyncSession,
        tx_id: str,
    ) -> Sequence[AnalystFeedback]:
        """Retrieves all feedback records submitted for a specific transaction."""
        query = (
            select(AnalystFeedback)
            .where(AnalystFeedback.tx_id == tx_id)
            .order_by(AnalystFeedback.reviewed_at.desc())
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def get_by_analyst(
        self,
        session: AsyncSession,
        analyst_id: str,
        limit: int = 50,
    ) -> Sequence[AnalystFeedback]:
        """Retrieves audit trail of feedback submitted by a specific reviewer."""
        query = (
            select(AnalystFeedback)
            .where(AnalystFeedback.analyst_id == analyst_id)
            .order_by(AnalystFeedback.reviewed_at.desc())
            .limit(limit)
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def get_feedback_statistics(self, session: AsyncSession) -> Dict[str, int]:
        """Calculates aggregated counts of verdicts across the database."""
        query = (
            select(AnalystFeedback.verdict, func.count(AnalystFeedback.id))
            .group_by(AnalystFeedback.verdict)
        )
        result = await session.execute(query)
        stats = {row[0]: row[1] for row in result.all()}
        return stats
