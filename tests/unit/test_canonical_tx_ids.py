"""Regression tests verifying canonical Elliptic transaction ID resolution across all endpoints."""

from typing import AsyncGenerator
import httpx
import pytest
import pytest_asyncio
import torch
from torch_geometric.data import Data
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.module3_graph_builder import get_graph_tx_ids
from src.module6_api.dependencies import get_db
from src.module6_api.main import app
from src.module8_database.models.base import Base


def test_get_graph_tx_ids_helper():
    """Verifies get_graph_tx_ids prioritizes tx_id, supports node_tx_ids, and fails on empty."""
    # 1. Standard PyG graph with tx_id
    d1 = Data(x=torch.zeros((3, 10)), edge_index=torch.empty((2, 0), dtype=torch.long), tx_id=["tx_A", "tx_B", "tx_C"])
    assert get_graph_tx_ids(d1) == ["tx_A", "tx_B", "tx_C"]

    # 2. Backwards-compatible node_tx_ids
    d2 = Data(x=torch.zeros((2, 10)), edge_index=torch.empty((2, 0), dtype=torch.long))
    d2.node_tx_ids = ["tx_X", "tx_Y"]
    assert get_graph_tx_ids(d2) == ["tx_X", "tx_Y"]

    # 3. Missing both -> raises ValueError (never silently creates numeric indices)
    d3 = Data(x=torch.zeros((2, 10)), edge_index=torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="Graph Data object does not contain canonical"):
        get_graph_tx_ids(d3)


@pytest_asyncio.fixture
async def empty_db_api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an isolated API client where DB has zero transactions to test artifact fallback."""
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

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    await test_engine.dispose()


@pytest.mark.asyncio
async def test_transactions_endpoint_returns_canonical_ids_not_node_indices(empty_db_api_client: httpx.AsyncClient):
    """
    Verifies that /api/v1/transactions returns actual Elliptic string transaction IDs
    (e.g., '97023208') rather than numeric indices like '0', '1', '1618'.
    """
    response = await empty_db_api_client.get("/api/v1/transactions?timestep=40&page=1&page_size=20")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    returned_ids = [item["tx_id"] for item in data["items"]]

    # Ensure no items are merely numeric index strings from range(0, 20)
    for idx_str in ["0", "1", "2", "1618", "1799", "3381", "3378"]:
        assert idx_str not in returned_ids, f"Found numeric index '{idx_str}' in returned transactions!"

    # Ensure returned IDs are valid canonical IDs (e.g. 7-10 digit numbers as strings)
    for tx_id in returned_ids:
        assert len(tx_id) >= 6, f"Expected canonical Elliptic tx_id, got: {tx_id}"

    # Verify search filtering on canonical transaction ID 10487903
    search_res = await empty_db_api_client.get("/api/v1/transactions?timestep=40&search=10487903")
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total"] >= 1
    assert any(item["tx_id"] == "10487903" for item in search_data["items"])


@pytest.mark.asyncio
async def test_canonical_graph_and_xai_lookup_succeeds(empty_db_api_client: httpx.AsyncClient):
    """
    Verifies that graph and XAI endpoints successfully locate real transactions by canonical ID.
    """
    canonical_tx = "230425980" # Timestep 1 canonical ID

    # 1. Graph lookup
    graph_res = await empty_db_api_client.get(f"/api/v1/graph/{canonical_tx}/subgraph?hops=2")
    assert graph_res.status_code == 200
    graph_data = graph_res.json()
    assert graph_data["target_tx_id"] == canonical_tx
    assert len(graph_data["nodes"]) >= 1

    target_node = next((n for n in graph_data["nodes"] if n["is_target"]), None)
    assert target_node is not None
    assert target_node["predicted_prob"] is not None
    assert isinstance(target_node["predicted_prob"], float)
    assert 0.0 <= target_node["predicted_prob"] <= 1.0
    assert target_node["risk_level"] == "LOW"

    # 2. XAI explanation lookup
    xai_res = await empty_db_api_client.get(f"/api/v1/explain/{canonical_tx}")
    assert xai_res.status_code == 200
    xai_data = xai_res.json()
    assert xai_data["tx_id"] == canonical_tx
    assert xai_data["prediction_probability"] is not None
    assert round(target_node["predicted_prob"], 4) == round(xai_data["prediction_probability"], 4)


@pytest.mark.asyncio
async def test_nonexistent_and_invalid_numeric_indices_return_404(empty_db_api_client: httpx.AsyncClient):
    """
    Verifies that querying a numeric index string that is NOT a canonical tx_id returns 404.
    """
    # '1618' is a node index, NOT a canonical transaction ID
    graph_res = await empty_db_api_client.get("/api/v1/graph/1618/subgraph")
    assert graph_res.status_code == 404

    xai_res = await empty_db_api_client.get("/api/v1/explain/1618")
    assert xai_res.status_code == 404

    detail_res = await empty_db_api_client.get("/api/v1/transactions/1618")
    assert detail_res.status_code == 404


@pytest.mark.asyncio
async def test_search_canonical_tx_10487903_variations(empty_db_api_client: httpx.AsyncClient):
    """
    Verifies searching by full ID (10487903), partial ID (1048), cross-timestep (230425980),
    empty search, and non-existent IDs.
    """
    # 1. Full search without specifying timestep
    res_full = await empty_db_api_client.get("/api/v1/transactions?search=10487903")
    assert res_full.status_code == 200
    data_full = res_full.json()
    assert data_full["total"] >= 1
    assert any(item["tx_id"] == "10487903" for item in data_full["items"])

    # 2. Partial search (1048)
    res_partial = await empty_db_api_client.get("/api/v1/transactions?search=1048")
    assert res_partial.status_code == 200
    data_partial = res_partial.json()
    assert data_partial["total"] >= 1
    assert any("1048" in item["tx_id"] for item in data_partial["items"])

    # 3. Cross-timestep search (230425980 in timestep 1 when timestep param is omitted)
    res_cross = await empty_db_api_client.get("/api/v1/transactions?search=230425980")
    assert res_cross.status_code == 200
    data_cross = res_cross.json()
    assert data_cross["total"] >= 1
    assert any(item["tx_id"] == "230425980" for item in data_cross["items"])

    # 4. Non-existent transaction search
    res_none = await empty_db_api_client.get("/api/v1/transactions?search=nonexistent_999999999")
    assert res_none.status_code == 200
    data_none = res_none.json()
    assert data_none["total"] == 0
    assert len(data_none["items"]) == 0


@pytest.mark.asyncio
async def test_canonical_10487903_xai_fidelity_and_benchmark_summary(empty_db_api_client: httpx.AsyncClient):
    """
    Verifies that explain endpoint returns non-null instance fidelity and canonical EXP-05 benchmark summary.
    """
    # 1. Explanation on canonical 10487903
    res = await empty_db_api_client.get("/api/v1/explain/10487903?timestep=40")
    assert res.status_code == 200
    expl = res.json()
    assert expl["tx_id"] == "10487903"
    assert expl["fidelity_plus"] is not None
    assert isinstance(expl["fidelity_plus"], float)
    assert expl["fidelity_minus"] is not None
    assert isinstance(expl["fidelity_minus"], float)
    assert expl["edge_sparsity"] is not None
    assert expl["feature_sparsity"] is not None
    assert expl["benchmark_summary"] is not None
    assert "gnn_explainer" in expl["benchmark_summary"]["overall_fidelity"]

    # 2. Benchmark summary endpoint
    bmk_res = await empty_db_api_client.get("/api/v1/explain/benchmark/summary")
    assert bmk_res.status_code == 200
    bmk = bmk_res.json()
    assert "overall_fidelity" in bmk
    assert "gnn_explainer" in bmk["overall_fidelity"]
    assert "gat_attention" in bmk["overall_fidelity"]
    assert "random" in bmk["overall_fidelity"]
    assert bmk["overall_fidelity"]["gnn_explainer"]["mean_fidelity_plus"] == 0.147
    assert bmk["overall_fidelity"]["gnn_explainer"]["mean_fidelity_minus"] == 0.0069
