"""Common response models and pagination wrappers."""

from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class HealthResponse(BaseModel):
    """Health check status payload."""
    status: str = "healthy"
    database_connected: bool
    model_available: bool
    active_model_version: Optional[str] = None
    architecture: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard generic wrapper for paginated entity lists."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class APIErrorResponse(BaseModel):
    """Standard structured error response schema."""
    error_code: str
    message: str
    details: Optional[Any] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
