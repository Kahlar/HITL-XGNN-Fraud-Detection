"""Unit and integration tests for Module 6 — FastAPI Async Backend Service."""

from typing import AsyncGenerator
import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.module6_api.dependencies import get_db
from src.module6_api.main import app
from src.module8_database.models.base import Base
from src.module8_database.models.transaction import Transaction


@pytest_asyncio.fixture
async def async_api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an isolated test HTTP client with an in-memory SQLite database session override."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Seed sample transactions in DB
    async with session_factory() as session:
        tx1 = Transaction(
            tx_id="tx_test_101",
            timestep=40,
            ground_truth_label=1,
            predicted_prob=0.88,
            predicted_class=1,
            risk_level="CRITICAL",
            uncertainty_score=0.40,
            entropy=0.52,
            priority_score=0.86,
            triage_status="QUEUED",
        )
        tx2 = Transaction(
            tx_id="tx_test_102",
            timestep=40,
            ground_truth_label=0,
            predicted_prob=0.54,
            predicted_class=0,
            risk_level="MEDIUM",
            uncertainty_score=0.98,
            entropy=0.99,
            priority_score=0.91,
            triage_status="QUEUED",
        )
        session.add_all([tx1, tx2])
        await session.commit()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    await test_engine.dispose()


@pytest.mark.asyncio
async def test_health_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/health returns readiness status and model info."""
    response = await async_api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["database_connected"] is True
    assert "active_model_version" in data


@pytest.mark.asyncio
async def test_list_transactions_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/transactions returns paginated list and filters by search."""
    # 1. Unfiltered list
    response = await async_api_client.get("/api/v1/transactions?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) >= 2
    assert data["items"][0]["tx_id"] in ("tx_test_101", "tx_test_102")

    # 2. Search exact match
    search_res = await async_api_client.get("/api/v1/transactions?search=tx_test_101")
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert s_data["total"] == 1
    assert len(s_data["items"]) == 1
    assert s_data["items"][0]["tx_id"] == "tx_test_101"

    # 3. Search substring match
    substr_res = await async_api_client.get("/api/v1/transactions?search=102")
    assert substr_res.status_code == 200
    sub_data = substr_res.json()
    assert sub_data["total"] == 1
    assert len(sub_data["items"]) == 1
    assert sub_data["items"][0]["tx_id"] == "tx_test_102"

    # 4. Search no match
    none_res = await async_api_client.get("/api/v1/transactions?search=nonexistent_tx_xyz")
    assert none_res.status_code == 200
    none_data = none_res.json()
    assert none_data["total"] == 0
    assert len(none_data["items"]) == 0


@pytest.mark.asyncio
async def test_get_transaction_detail(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/transactions/{tx_id} for valid and invalid transactions."""
    # Valid transaction from DB
    res_valid = await async_api_client.get("/api/v1/transactions/tx_test_101")
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["tx_id"] == "tx_test_101"
    assert data["risk_level"] == "CRITICAL"

    # Non-existent transaction
    res_invalid = await async_api_client.get("/api/v1/transactions/non_existent_tx_99999")
    assert res_invalid.status_code == 404


@pytest.mark.asyncio
async def test_subgraph_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/graph/{tx_id}/subgraph returns valid nodes, edges, and numeric predicted_prob."""
    # Sample a real transaction from timestep 1
    response = await async_api_client.get("/api/v1/graph/230425980/subgraph?hops=2&timestep=1")
    assert response.status_code == 200
    data = response.json()
    assert data["target_tx_id"] == "230425980"
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 1

    # Verify target node has numeric non-null predicted_prob and valid risk tier
    target_node = next((n for n in data["nodes"] if n["is_target"]), data["nodes"][0])
    assert target_node["predicted_prob"] is not None
    assert isinstance(target_node["predicted_prob"], float)
    assert 0.0 <= target_node["predicted_prob"] <= 1.0
    assert target_node["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@pytest.mark.asyncio
async def test_explain_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/explain/{tx_id} returns attribution rankings and fidelity metrics."""
    response = await async_api_client.get("/api/v1/explain/230425980?timestep=1")
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == "230425980"
    assert "top_features" in data
    assert "top_edges" in data
    assert "feature_summary" in data


@pytest.mark.asyncio
async def test_hitl_feedback_submission(async_api_client: httpx.AsyncClient):
    """Verifies POST /api/v1/hitl/feedback records verdict and validates schema."""
    payload = {
        "tx_id": "tx_test_101",
        "analyst_id": "analyst_clara",
        "verdict": "ILLICIT",
        "confidence": 5,
        "rationale": "High-degree fan-out confirmed as mixing entity.",
        "explanation_viewed": True,
        "feedback_source": "HUMAN_ANALYST",
    }
    response = await async_api_client.post("/api/v1/hitl/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["tx_id"] == "tx_test_101"
    assert data["verdict"] == "ILLICIT"
    assert data["analyst_id"] == "analyst_clara"

    # Verify transaction triage_status updated
    tx_res = await async_api_client.get("/api/v1/transactions/tx_test_101")
    assert tx_res.json()["triage_status"] == "REVIEWED"

    # Test invalid verdict rejection
    bad_payload = payload.copy()
    bad_payload["verdict"] = "INVALID_VERDICT"
    bad_res = await async_api_client.post("/api/v1/hitl/feedback", json=bad_payload)
    assert bad_res.status_code in (422, 400)


@pytest.mark.asyncio
async def test_hitl_queue_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/hitl/queue returns prioritized triage items."""
    response = await async_api_client.get("/api/v1/hitl/queue?min_priority=0.50&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "risk_distribution" in data
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_analytics_metrics_endpoint(async_api_client: httpx.AsyncClient):
    """Verifies GET /api/v1/analytics/metrics returns dataset, active learning, and EXP-06 temporal metrics."""
    response = await async_api_client.get("/api/v1/analytics/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_summary" in data
    assert "active_model" in data
    assert "temporal_drift" in data
    assert "active_learning_benchmarks" in data
    assert data["dataset_summary"]["total_transactions"] == 203769

    # Verify EXP-06 temporal drift items
    drift = data["temporal_drift"]
    assert len(drift) == 10  # Timesteps 40..49
    ts40 = drift[0]
    assert ts40["timestep"] == 40
    assert ts40["period"] == "pre_shock"
    assert ts40["num_labeled"] == 1211
    assert ts40["num_illicit"] == 112
    assert ts40["illicit_prevalence_pct"] == 9.25
    assert round(ts40["f1_score"], 4) == 0.5952
    assert round(ts40["precision"], 4) == 0.8929
    assert round(ts40["recall"], 4) == 0.4464

    # Verify EXP-07 active learning metrics remain unaffected
    assert "f1_budget_table" in data["active_learning_benchmarks"]


@pytest.mark.asyncio
async def test_openapi_route_registration(async_api_client: httpx.AsyncClient):
    """Verifies OpenAPI schema reflects all 8 primary endpoints."""
    response = await async_api_client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})

    expected_routes = [
        "/api/v1/health",
        "/api/v1/transactions",
        "/api/v1/transactions/{tx_id}",
        "/api/v1/graph/{tx_id}/subgraph",
        "/api/v1/explain/{tx_id}",
        "/api/v1/hitl/feedback",
        "/api/v1/hitl/queue",
        "/api/v1/analytics/metrics",
    ]
    for route in expected_routes:
        assert route in paths, f"Route {route} not found in OpenAPI schema"
