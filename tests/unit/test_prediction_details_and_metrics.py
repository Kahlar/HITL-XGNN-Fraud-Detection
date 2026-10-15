"""Unit and integration tests for prediction details, group metrics, threshold policy, and model validation."""

from typing import AsyncGenerator
import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.module6_api.dependencies import get_db
from src.module6_api.main import app
from src.module6_api.routers.transactions import _determine_outcome_category
from src.module8_database.models.base import Base
from src.module8_database.models.transaction import Transaction


def test_determine_outcome_category():
    """Verifies standard 2x2 confusion outcome mapping and unlabelled handling."""
    # TP
    assert _determine_outcome_category(ground_truth_label=1, predicted_class=1) == "TRUE_POSITIVE"
    # FP
    assert _determine_outcome_category(ground_truth_label=0, predicted_class=1) == "FALSE_POSITIVE"
    # TN
    assert _determine_outcome_category(ground_truth_label=0, predicted_class=0) == "TRUE_NEGATIVE"
    # FN
    assert _determine_outcome_category(ground_truth_label=1, predicted_class=0) == "FALSE_NEGATIVE"
    # Unlabeled / masked
    assert _determine_outcome_category(ground_truth_label=-1, predicted_class=1) == "NOT_EVALUABLE"
    assert _determine_outcome_category(ground_truth_label=-1, predicted_class=0) == "NOT_EVALUABLE"
    assert _determine_outcome_category(ground_truth_label=2, predicted_class=0) == "NOT_EVALUABLE"
    assert _determine_outcome_category(ground_truth_label=1, predicted_class=None) == "NOT_EVALUABLE"


@pytest_asyncio.fixture
async def test_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an isolated test client with an in-memory SQLite database."""
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
async def test_canonical_tx_prediction_details(test_client: httpx.AsyncClient):
    """Verifies that GET /api/v1/transactions/10487903 exposes rich prediction telemetry."""
    res = await test_client.get("/api/v1/transactions/10487903")
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["tx_id"] == "10487903"
    assert data["timestep"] == 40
    assert data["model_version"] in ("graphsage_hitl_uncertainty_20", "graphsage_base")
    assert data["architecture"] == "GraphSAGE"
    assert data["decision_threshold"] == 0.5517
    assert "0.5517" in data["decision_rule"]
    assert data["predicted_prob"] is not None
    assert data["predicted_prob"] >= 0.5517
    assert data["predicted_class"] == 1
    # Canonical tx 10487903 is unlabelled in raw Elliptic dataset (-1)
    assert data["ground_truth_label"] == -1
    assert data["ground_truth_status"] == "unavailable"
    assert data["outcome_category"] == "NOT_EVALUABLE"
    assert data["uncertainty_score"] is not None
    assert data["entropy"] is not None
    assert data["explanation_method"] == "GNNExplainer"


@pytest.mark.asyncio
async def test_labeled_tx_prediction_details(test_client: httpx.AsyncClient):
    """Verifies that known labeled transactions in TS 40 yield correct outcome classifications."""
    # 1. True Positive: illicit node with prob >= 0.5517
    res_tp = await test_client.get("/api/v1/transactions/96925342")
    assert res_tp.status_code == 200, res_tp.text
    data_tp = res_tp.json()
    assert data_tp["ground_truth_label"] == 1
    assert data_tp["predicted_class"] == 1
    assert data_tp["outcome_category"] == "TRUE_POSITIVE"

    # 2. False Negative: illicit node with prob < 0.5517
    res_fn = await test_client.get("/api/v1/transactions/97011545")
    assert res_fn.status_code == 200, res_fn.text
    data_fn = res_fn.json()
    assert data_fn["ground_truth_label"] == 1
    assert data_fn["predicted_class"] == 0
    assert data_fn["outcome_category"] == "FALSE_NEGATIVE"

    # 3. True Negative: licit node with prob < 0.5517
    res_tn = await test_client.get("/api/v1/transactions/97425476")
    assert res_tn.status_code == 200, res_tn.text
    data_tn = res_tn.json()
    assert data_tn["ground_truth_label"] == 0
    assert data_tn["predicted_class"] == 0
    assert data_tn["outcome_category"] == "TRUE_NEGATIVE"


@pytest.mark.asyncio
async def test_threshold_policy_endpoint(test_client: httpx.AsyncClient):
    """Verifies GET /api/v1/models/threshold returns calibration methodology and comparisons."""
    res = await test_client.get("/api/v1/models/threshold")
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["calibrated_threshold"] == 0.5517
    assert data["architecture"] == "GraphSAGE"
    assert "Grid Search" in data["calibration_method"]
    assert "35, 39" in data["calibration_split"]
    assert data["grid_specification"]["num_points"] == 200
    assert data["comparison_thresholds"]["xgboost"] == 0.9014
    assert data["comparison_thresholds"]["lightgbm"] == 0.8915
    assert data["comparison_thresholds"]["graphsage"] == 0.5517
    assert "binary_classification" in data["decision_rules"]


@pytest.mark.asyncio
async def test_model_validation_endpoint(test_client: httpx.AsyncClient):
    """Verifies GET /api/v1/models/validation returns 7-model inventory and zero leakage metadata."""
    res = await test_client.get("/api/v1/models/validation")
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["train_timesteps"] == [1, 34]
    assert data["validation_timesteps"] == [35, 39]
    assert data["test_timesteps"] == [40, 49]
    assert "zero" in data["zero_leakage_policy"].lower()

    models = data["models"]
    assert len(models) >= 5
    model_names = [m["name"].lower() for m in models]
    assert any("xgboost" in name for name in model_names)
    assert any("graphsage" in name for name in model_names)

    evidence = data["verification_evidence"]
    assert evidence["verification_status"] == "VERIFIED"
    assert evidence["dataset_invariants"]["zero_orphan_nodes"] is True
    assert evidence["dataset_invariants"]["intra_timestep_edges_pct"] == 100.0


@pytest.mark.asyncio
async def test_group_evaluation_timestep_scope(test_client: httpx.AsyncClient):
    """Verifies group metric evaluation on timestep 40 masks unlabelled nodes and computes F1/PR-AUC."""
    payload = {"scope": "timestep", "timestep": 40}
    res = await test_client.post("/api/v1/analytics/evaluation-metrics", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["scope"] == "timestep"
    assert data["timestep"] == 40
    assert data["total_transactions"] > 0
    assert data["labeled_count"] > 0
    assert data["unlabeled_count"] > 0
    assert data["total_transactions"] == data["labeled_count"] + data["unlabeled_count"]
    assert data["decision_threshold"] == 0.5517

    # Metric bounds
    assert 0.0 <= data["f1_score"] <= 1.0
    assert 0.0 <= data["pr_auc"] <= 1.0
    assert 0.0 <= data["precision"] <= 1.0
    assert 0.0 <= data["recall"] <= 1.0
    assert data["true_positives"] + data["false_negatives"] == data["illicit_count"]
    assert data["true_negatives"] + data["false_positives"] == data["licit_count"]
    assert data["status"] == "VALID"


@pytest.mark.asyncio
async def test_group_evaluation_filtered_scope(test_client: httpx.AsyncClient):
    """Verifies group metric evaluation on filtered subsets: labeled tx vs unlabelled tx."""
    # 1. Labeled illicit transaction (True Positive)
    payload_labeled = {"scope": "filtered", "timestep": 40, "tx_ids": ["96925342"]}
    res = await test_client.post("/api/v1/analytics/evaluation-metrics", json=payload_labeled)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["scope"] == "filtered"
    assert data["total_transactions"] == 1
    assert data["labeled_count"] == 1
    assert data["illicit_count"] == 1
    assert data["true_positives"] == 1

    # 2. Unlabelled transaction (masked from supervised evaluation)
    payload_unlabeled = {"scope": "filtered", "timestep": 40, "tx_ids": ["10487903"]}
    res2 = await test_client.post("/api/v1/analytics/evaluation-metrics", json=payload_unlabeled)
    assert res2.status_code == 200, res2.text
    data2 = res2.json()

    assert data2["scope"] == "filtered"
    assert data2["total_transactions"] == 1
    assert data2["labeled_count"] == 0
    assert data2["unlabeled_count"] == 1
    assert data2["status"] == "INSUFFICIENT_LABELED_DATA"
