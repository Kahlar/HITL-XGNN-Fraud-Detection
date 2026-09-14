"""Repository for transaction entities, triage queries, and graph payment flows."""

from typing import List, Optional, Sequence, Set
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.module8_database.models.transaction import Transaction, TransactionEdge
from src.module8_database.repositories.base_repository import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    """Data access repository for Transactions and TransactionEdges."""

    def __init__(self):
        super().__init__(Transaction)

    async def get_by_tx_id(
        self,
        session: AsyncSession,
        tx_id: str,
        load_details: bool = True,
    ) -> Optional[Transaction]:
        """Fetches a transaction, optionally eager loading all relationships."""
        if not load_details:
            return await session.get(Transaction, tx_id)

        query = (
            select(Transaction)
            .where(Transaction.tx_id == tx_id)
            .options(
                selectinload(Transaction.explanations),
                selectinload(Transaction.feedbacks),
                selectinload(Transaction.outgoing_edges),
                selectinload(Transaction.incoming_edges),
            )
        )
        result = await session.execute(query)
        return result.scalars().first()

    def _build_triage_filters(
        self,
        timestep: Optional[int] = None,
        risk_levels: Optional[List[str]] = None,
        min_priority: float = 0.0,
        triage_status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list:
        filters = [Transaction.priority_score >= min_priority]

        if timestep is not None:
            filters.append(Transaction.timestep == timestep)
        if risk_levels:
            filters.append(Transaction.risk_level.in_(risk_levels))
        if triage_status:
            filters.append(Transaction.triage_status == triage_status)
        if search and search.strip():
            filters.append(Transaction.tx_id.ilike(f"%{search.strip()}%"))

        return filters

    async def get_triage_queue(
        self,
        session: AsyncSession,
        timestep: Optional[int] = None,
        risk_levels: Optional[List[str]] = None,
        min_priority: float = 0.0,
        triage_status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Transaction]:
        """
        Queries prioritized transactions requiring review, sorted descending by priority_score.
        """
        filters = self._build_triage_filters(
            timestep=timestep,
            risk_levels=risk_levels,
            min_priority=min_priority,
            triage_status=triage_status,
            search=search,
        )

        query = (
            select(Transaction)
            .where(and_(*filters))
            .order_by(Transaction.priority_score.desc().nullslast())
            .limit(limit)
            .offset(offset)
            .options(selectinload(Transaction.feedbacks))
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def count_triage_queue(
        self,
        session: AsyncSession,
        timestep: Optional[int] = None,
        risk_levels: Optional[List[str]] = None,
        min_priority: float = 0.0,
        triage_status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> int:
        """Counts transactions matching the triage queue filters."""
        filters = self._build_triage_filters(
            timestep=timestep,
            risk_levels=risk_levels,
            min_priority=min_priority,
            triage_status=triage_status,
            search=search,
        )
        query = select(func.count()).select_from(Transaction).where(and_(*filters))
        result = await session.execute(query)
        return int(result.scalar_one())

    async def get_subgraph_edges(
        self,
        session: AsyncSession,
        tx_ids: List[str],
        timestep: Optional[int] = None,
    ) -> Sequence[TransactionEdge]:
        """Retrieves all directed payment edges connecting the given set of transaction IDs."""
        if not tx_ids:
            return []

        filters = [
            TransactionEdge.source_tx_id.in_(tx_ids),
            TransactionEdge.target_tx_id.in_(tx_ids),
        ]
        if timestep is not None:
            filters.append(TransactionEdge.timestep == timestep)

        query = select(TransactionEdge).where(and_(*filters))
        result = await session.execute(query)
        return result.scalars().all()

    async def create_edges_batch(
        self,
        session: AsyncSession,
        edges: List[TransactionEdge],
    ) -> List[TransactionEdge]:
        """Batch inserts directed edges into the database."""
        session.add_all(edges)
        await session.flush()
        return edges
