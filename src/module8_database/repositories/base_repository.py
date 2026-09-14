"""Generic asynchronous repository with standard CRUD operations."""

from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.module8_database.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository implementing asynchronous CRUD accessors."""

    def __init__(self, model_class: Type[ModelType]):
        self.model_class = model_class

    async def get_by_id(self, session: AsyncSession, entity_id: Any) -> Optional[ModelType]:
        """Fetches a single entity by its primary key."""
        return await session.get(self.model_class, entity_id)

    async def get_all(
        self,
        session: AsyncSession,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ModelType]:
        """Retrieves a paginated sequence of entities."""
        query = select(self.model_class).limit(limit).offset(offset)
        result = await session.execute(query)
        return result.scalars().all()

    async def create(self, session: AsyncSession, entity: ModelType) -> ModelType:
        """Adds a single entity to the session."""
        session.add(entity)
        await session.flush()
        return entity

    async def create_batch(self, session: AsyncSession, entities: List[ModelType]) -> List[ModelType]:
        """Adds multiple entities in batch."""
        session.add_all(entities)
        await session.flush()
        return entities

    async def delete(self, session: AsyncSession, entity_id: Any) -> bool:
        """Deletes an entity by primary key, returning True if deleted."""
        entity = await self.get_by_id(session, entity_id)
        if entity:
            await session.delete(entity)
            await session.flush()
            return True
        return False

    async def count(self, session: AsyncSession) -> int:
        """Returns the total row count for the model table."""
        query = select(func.count()).select_from(self.model_class)
        result = await session.execute(query)
        return int(result.scalar_one())
