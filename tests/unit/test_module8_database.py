"""Unit and integration tests for Module 8 — PostgreSQL Database Integration & Feedback Store."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.module8_database.config import DatabaseSettings
from src.module8_database.models.base import Base
from src.module8_database.models.explanation import TransactionExplanation
from src.module8_database.models.feedback import AnalystFeedback
from src.module8_database.models.model_registry import ModelRegistry
from src.module8_database.models.transaction import Transaction, TransactionEdge
from src.module8_database.repositories.explanation_repo import ExplanationRepository
from src.module8_database.repositories.feedback_repo import FeedbackRepository
from src.module8_database.repositories.model_repo import ModelRegistryRepository
from src.module8_database.repositories.transaction_repo import TransactionRepository
from src.module8_database.service import DatabaseService


@pytest_asyncio.fixture
async def async_test_session():
    """Provides an isolated in-memory SQLite async session with fully initialized schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session

    await engine.dispose()


def test_database_configuration():
    """Verifies default settings and URL generation."""
    settings = DatabaseSettings(
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_DB="test_fraud_db",
    )
    assert "postgresql+asyncpg://test_user:test_password@localhost:5432/test_fraud_db" == settings.async_database_url
    assert "postgresql://test_user:test_password@localhost:5432/test_fraud_db" == settings.sync_database_url


def test_schema_metadata_integrity():
    """Verifies that all five core tables are declared in Base.metadata with proper columns."""
    tables = Base.metadata.tables
    expected_tables = {
        "transactions",
        "transaction_edges",
        "transaction_explanations",
        "analyst_feedback",
        "model_registry",
    }
    assert expected_tables.issubset(set(tables.keys()))
    assert "tx_id" in tables["transactions"].columns
    assert "priority_score" in tables["transactions"].columns
    assert "model_version" in tables["model_registry"].columns
    assert "verdict" in tables["analyst_feedback"].columns


@pytest.mark.asyncio
async def test_transaction_crud_and_triage_query(async_test_session: AsyncSession):
    """Verifies CRUD operations and prioritized triage queue querying."""
    repo = TransactionRepository()

    tx1 = Transaction(
        tx_id="tx_1001",
        timestep=35,
        ground_truth_label=1,
        predicted_prob=0.89,
        predicted_class=1,
        risk_level="CRITICAL",
        uncertainty_score=0.35,
        entropy=0.50,
        priority_score=0.85,
        is_triaged=False,
        triage_status="QUEUED",
    )
    tx2 = Transaction(
        tx_id="tx_1002",
        timestep=35,
        ground_truth_label=0,
        predicted_prob=0.55,
        predicted_class=0,
        risk_level="MEDIUM",
        uncertainty_score=0.99,
        entropy=0.99,
        priority_score=0.92,
        is_triaged=False,
        triage_status="QUEUED",
    )
    await repo.create(async_test_session, tx1)
    await repo.create(async_test_session, tx2)
    await async_test_session.commit()

    # Query by ID
    fetched = await repo.get_by_tx_id(async_test_session, "tx_1001")
    assert fetched is not None
    assert fetched.tx_id == "tx_1001"
    assert fetched.risk_level == "CRITICAL"

    # Query triage queue sorted descending by priority_score
    queue = await repo.get_triage_queue(async_test_session, timestep=35, min_priority=0.50)
    assert len(queue) == 2
    # tx_1002 has priority 0.92, tx_1001 has 0.85
    assert queue[0].tx_id == "tx_1002"
    assert queue[1].tx_id == "tx_1001"


@pytest.mark.asyncio
async def test_transaction_edges_and_subgraph_query(async_test_session: AsyncSession):
    """Verifies directed graph payment edge insertion and subgraph connectivity queries."""
    tx_repo = TransactionRepository()

    # Insert transactions
    tx_a = Transaction(tx_id="tx_A", timestep=35, ground_truth_label=1)
    tx_b = Transaction(tx_id="tx_B", timestep=35, ground_truth_label=0)
    tx_c = Transaction(tx_id="tx_C", timestep=35, ground_truth_label=-1)
    await tx_repo.create(async_test_session, tx_a)
    await tx_repo.create(async_test_session, tx_b)
    await tx_repo.create(async_test_session, tx_c)

    # Insert edges: A -> B, B -> C
    edge1 = TransactionEdge(source_tx_id="tx_A", target_tx_id="tx_B", timestep=35)
    edge2 = TransactionEdge(source_tx_id="tx_B", target_tx_id="tx_C", timestep=35)
    await tx_repo.create_edges_batch(async_test_session, [edge1, edge2])
    await async_test_session.commit()

    # Query subgraph edges connecting ['tx_A', 'tx_B', 'tx_C']
    subgraph_edges = await tx_repo.get_subgraph_edges(async_test_session, ["tx_A", "tx_B", "tx_C"], timestep=35)
    assert len(subgraph_edges) == 2
    assert {(e.source_tx_id, e.target_tx_id) for e in subgraph_edges} == {("tx_A", "tx_B"), ("tx_B", "tx_C")}


@pytest.mark.asyncio
async def test_explanation_repository_storage(async_test_session: AsyncSession):
    """Verifies storing and retrieving GNNExplainer attribution records and JSON payloads."""
    tx_repo = TransactionRepository()
    expl_repo = ExplanationRepository()

    tx = Transaction(tx_id="tx_expl_01", timestep=40, ground_truth_label=1)
    await tx_repo.create(async_test_session, tx)

    explanation = TransactionExplanation(
        tx_id="tx_expl_01",
        model_version="graphsage_hitl_uncertainty_20",
        fidelity_plus=0.3450,
        fidelity_minus=-0.0120,
        edge_sparsity=0.7938,
        feature_sparsity=0.9000,
        generation_latency_ms=285.5,
        edge_attributions=[{"source": "tx_expl_01", "target": "tx_99", "importance": 0.85}],
        feature_attributions=[{"feature_idx": 42, "importance": 0.92, "type": "local"}],
        subgraph_nodes=["tx_expl_01", "tx_99"],
        subgraph_edges=[["tx_expl_01", "tx_99"]],
        feature_summary={"dominant_feature_type": "local"},
    )
    await expl_repo.create(async_test_session, explanation)
    await async_test_session.commit()

    fetched = await expl_repo.get_latest_by_tx(async_test_session, "tx_expl_01")
    assert fetched is not None
    assert fetched.model_version == "graphsage_hitl_uncertainty_20"
    assert fetched.fidelity_plus == 0.3450
    assert fetched.edge_attributions[0]["importance"] == 0.85


@pytest.mark.asyncio
async def test_analyst_feedback_submission_and_status_transition(async_test_session: AsyncSession):
    """Verifies reviewer submission and automatic parent transaction status transition to REVIEWED."""
    tx_repo = TransactionRepository()
    fb_repo = FeedbackRepository()

    tx = Transaction(tx_id="tx_fb_01", timestep=36, ground_truth_label=1, triage_status="QUEUED")
    await tx_repo.create(async_test_session, tx)
    await async_test_session.commit()

    feedback = AnalystFeedback(
        tx_id="tx_fb_01",
        analyst_id="analyst_smith",
        verdict="ILLICIT",
        confidence=5,
        rationale="Clear multi-hop tumbling pattern corroborated by XAI subgraph.",
        model_predicted_score=0.91,
        explanation_viewed=True,
        feedback_source="HUMAN_ANALYST",
        reviewed_at=datetime.now(timezone.utc),
    )
    await fb_repo.submit_feedback(async_test_session, feedback)
    await async_test_session.commit()

    # Verify feedback record
    records = await fb_repo.get_by_tx(async_test_session, "tx_fb_01")
    assert len(records) == 1
    assert records[0].verdict == "ILLICIT"
    assert records[0].analyst_id == "analyst_smith"

    # Verify transaction triage_status updated to REVIEWED
    updated_tx = await tx_repo.get_by_tx_id(async_test_session, "tx_fb_01")
    assert updated_tx.triage_status == "REVIEWED"


@pytest.mark.asyncio
async def test_model_registry_versioning_and_active_toggle(async_test_session: AsyncSession):
    """Verifies model registration, metric tracking, and active deployment toggle."""
    model_repo = ModelRegistryRepository()

    m1 = ModelRegistry(
        model_version="graphsage_base",
        architecture="GraphSAGE",
        feature_set="original_all",
        feature_count=165,
        loss_function="focal",
        decision_threshold=0.5517,
        test_f1=0.5175,
        test_pr_auc=0.4499,
        checkpoint_path="models/gnn/graphsage_best.pt",
        is_active_for_inference=True,
    )
    m2 = ModelRegistry(
        model_version="graphsage_hitl_uncertainty_20",
        base_model_version="graphsage_base",
        architecture="GraphSAGE",
        feature_set="original_all",
        feature_count=165,
        loss_function="focal",
        decision_threshold=0.5517,
        feedback_strategy="uncertainty",
        feedback_budget_ratio=0.20,
        feedback_sample_count=1002,
        test_f1=0.5030,
        test_pr_auc=0.4358,
        checkpoint_path="models/gnn/hitl/graphsage_hitl_uncertainty_20.pt",
        is_active_for_inference=False,
    )
    await model_repo.register_model(async_test_session, m1)
    await model_repo.register_model(async_test_session, m2)
    await async_test_session.commit()

    # Active model initially m1
    active = await model_repo.get_active_model(async_test_session, architecture="GraphSAGE")
    assert active.model_version == "graphsage_base"

    # Switch active model to m2
    updated = await model_repo.set_active_model(async_test_session, "graphsage_hitl_uncertainty_20")
    await async_test_session.commit()
    assert updated.is_active_for_inference is True

    # Check that m1 was deactivated
    m1_refetched = await model_repo.get_by_id(async_test_session, "graphsage_base")
    assert m1_refetched.is_active_for_inference is False


@pytest.mark.asyncio
async def test_database_service_seeding_from_exp07(async_test_session: AsyncSession):
    """Verifies seeding model registry from historical EXP-07 active learning JSON artifact."""
    service = DatabaseService()
    exp07_path = Path("data/processed/experiments/exp07_hitl_learning.json")

    if exp07_path.exists():
        count = await service.seed_model_registry_from_exp07(async_test_session, exp07_path)
        await async_test_session.commit()
        assert count == 15

        all_models = await service.model_repo.get_all_models(async_test_session)
        assert len(all_models) == 15

        active_model = await service.model_repo.get_active_model(async_test_session)
        assert active_model is not None
        assert active_model.model_version == "graphsage_hitl_uncertainty_20"
