"""Module 8 — PostgreSQL Database Integration & Feedback Store."""

from src.module8_database.config import DatabaseSettings, db_settings
from src.module8_database.connection import DatabaseSessionManager, db_manager, get_db_session
from src.module8_database.models import (
    AnalystFeedback,
    Base,
    ModelRegistry,
    TimestampMixin,
    Transaction,
    TransactionEdge,
    TransactionExplanation,
)
from src.module8_database.repositories import (
    BaseRepository,
    ExplanationRepository,
    FeedbackRepository,
    ModelRegistryRepository,
    TransactionRepository,
)
from src.module8_database.service import DatabaseService

__all__ = [
    "db_settings",
    "DatabaseSettings",
    "db_manager",
    "DatabaseSessionManager",
    "get_db_session",
    "Base",
    "TimestampMixin",
    "Transaction",
    "TransactionEdge",
    "TransactionExplanation",
    "AnalystFeedback",
    "ModelRegistry",
    "BaseRepository",
    "TransactionRepository",
    "ExplanationRepository",
    "FeedbackRepository",
    "ModelRegistryRepository",
    "DatabaseService",
]
