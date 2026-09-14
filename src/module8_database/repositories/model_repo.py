"""Repository for model checkpoint registry, provenance, and active version management."""

from typing import List, Optional, Sequence
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.module8_database.models.model_registry import ModelRegistry
from src.module8_database.repositories.base_repository import BaseRepository


class ModelRegistryRepository(BaseRepository[ModelRegistry]):
    """Data access repository for ModelRegistry entries."""

    def __init__(self):
        super().__init__(ModelRegistry)

    async def register_model(
        self,
        session: AsyncSession,
        model_entry: ModelRegistry,
    ) -> ModelRegistry:
        """Registers a new model version or updates existing entry."""
        existing = await session.get(ModelRegistry, model_entry.model_version)
        if existing:
            for key, val in model_entry.__dict__.items():
                if not key.startswith("_") and val is not None:
                    setattr(existing, key, val)
            await session.flush()
            return existing
        else:
            session.add(model_entry)
            await session.flush()
            return model_entry

    async def get_active_model(
        self,
        session: AsyncSession,
        architecture: Optional[str] = None,
    ) -> Optional[ModelRegistry]:
        """Retrieves the currently active model version for online inference."""
        query = select(ModelRegistry).where(ModelRegistry.is_active_for_inference.is_(True))
        if architecture:
            query = query.where(ModelRegistry.architecture == architecture)
        result = await session.execute(query)
        return result.scalars().first()

    async def set_active_model(
        self,
        session: AsyncSession,
        model_version: str,
    ) -> Optional[ModelRegistry]:
        """
        Sets a model version as active for inference while deactivating others of the same architecture.
        """
        target = await session.get(ModelRegistry, model_version)
        if not target:
            return None

        # Deactivate previous active models of the same architecture
        deactivate_query = (
            update(ModelRegistry)
            .where(ModelRegistry.architecture == target.architecture)
            .values(is_active_for_inference=False)
        )
        await session.execute(deactivate_query)

        target.is_active_for_inference = True
        await session.flush()
        return target

    async def get_all_models(
        self,
        session: AsyncSession,
    ) -> Sequence[ModelRegistry]:
        """Retrieves all registered models ordered chronologically descending."""
        query = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
        result = await session.execute(query)
        return result.scalars().all()
