"""Initial database schema for HITL-XGNN Fraud Detection.

Revision ID: 001_initial_schema
Revises: None
Create Date: 2026-09-01 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create transactions table
    op.create_table(
        "transactions",
        sa.Column("tx_id", sa.String(length=64), nullable=False),
        sa.Column("timestep", sa.Integer(), nullable=False),
        sa.Column("ground_truth_label", sa.Integer(), nullable=False, server_default="-1"),
        sa.Column("predicted_prob", sa.Float(), nullable=True),
        sa.Column("predicted_class", sa.Integer(), nullable=True),
        sa.Column("risk_level", sa.String(length=32), nullable=True),
        sa.Column("uncertainty_score", sa.Float(), nullable=True),
        sa.Column("entropy", sa.Float(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("is_triaged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("triage_status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tx_id"),
    )
    op.create_index("ix_transactions_tx_id", "transactions", ["tx_id"])
    op.create_index("ix_transactions_timestep", "transactions", ["timestep"])
    op.create_index("ix_transactions_risk_level", "transactions", ["risk_level"])
    op.create_index("ix_transactions_priority_score", "transactions", ["priority_score"])
    op.create_index("ix_transactions_timestep_triaged", "transactions", ["timestep", "is_triaged"])
    op.create_index("ix_transactions_priority_risk", "transactions", ["priority_score", "risk_level"])

    # 2. Create transaction_edges table
    op.create_table(
        "transaction_edges",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_tx_id", sa.String(length=64), nullable=False),
        sa.Column("target_tx_id", sa.String(length=64), nullable=False),
        sa.Column("timestep", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_tx_id"], ["transactions.tx_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_tx_id"], ["transactions.tx_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_transaction_edges_source_tx_id", "transaction_edges", ["source_tx_id"])
    op.create_index("ix_transaction_edges_target_tx_id", "transaction_edges", ["target_tx_id"])
    op.create_index("ix_transaction_edges_timestep", "transaction_edges", ["timestep"])
    op.create_index("ix_edges_source_target_ts", "transaction_edges", ["source_tx_id", "target_tx_id", "timestep"])

    # 3. Create transaction_explanations table
    op.create_table(
        "transaction_explanations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tx_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("fidelity_plus", sa.Float(), nullable=True),
        sa.Column("fidelity_minus", sa.Float(), nullable=True),
        sa.Column("edge_sparsity", sa.Float(), nullable=True),
        sa.Column("feature_sparsity", sa.Float(), nullable=True),
        sa.Column("generation_latency_ms", sa.Float(), nullable=True),
        sa.Column("edge_attributions", sa.JSON(), nullable=True),
        sa.Column("feature_attributions", sa.JSON(), nullable=True),
        sa.Column("subgraph_nodes", sa.JSON(), nullable=True),
        sa.Column("subgraph_edges", sa.JSON(), nullable=True),
        sa.Column("feature_summary", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tx_id"], ["transactions.tx_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_transaction_explanations_tx_id", "transaction_explanations", ["tx_id"])
    op.create_index("ix_transaction_explanations_model_version", "transaction_explanations", ["model_version"])
    op.create_index("ix_explanations_tx_model", "transaction_explanations", ["tx_id", "model_version"])

    # 4. Create analyst_feedback table
    op.create_table(
        "analyst_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tx_id", sa.String(length=64), nullable=False),
        sa.Column("analyst_id", sa.String(length=64), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("model_predicted_score", sa.Float(), nullable=True),
        sa.Column("explanation_viewed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("feedback_source", sa.String(length=32), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tx_id"], ["transactions.tx_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyst_feedback_tx_id", "analyst_feedback", ["tx_id"])
    op.create_index("ix_analyst_feedback_analyst_id", "analyst_feedback", ["analyst_id"])
    op.create_index("ix_analyst_feedback_verdict", "analyst_feedback", ["verdict"])
    op.create_index("ix_feedback_analyst_verdict", "analyst_feedback", ["analyst_id", "verdict"])
    op.create_index("ix_feedback_reviewed_at", "analyst_feedback", ["reviewed_at"])

    # 5. Create model_registry table
    op.create_table(
        "model_registry",
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("base_model_version", sa.String(length=64), nullable=True),
        sa.Column("architecture", sa.String(length=32), nullable=False),
        sa.Column("feature_set", sa.String(length=32), nullable=False, server_default="original_all"),
        sa.Column("feature_count", sa.Integer(), nullable=False, server_default="165"),
        sa.Column("loss_function", sa.String(length=32), nullable=False, server_default="focal"),
        sa.Column("decision_threshold", sa.Float(), nullable=False, server_default="0.5517"),
        sa.Column("feedback_strategy", sa.String(length=32), nullable=True),
        sa.Column("feedback_budget_ratio", sa.Float(), nullable=True),
        sa.Column("feedback_sample_count", sa.Integer(), nullable=True),
        sa.Column("test_f1", sa.Float(), nullable=True),
        sa.Column("test_pr_auc", sa.Float(), nullable=True),
        sa.Column("test_precision", sa.Float(), nullable=True),
        sa.Column("test_recall", sa.Float(), nullable=True),
        sa.Column("test_roc_auc", sa.Float(), nullable=True),
        sa.Column("test_accuracy", sa.Float(), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=256), nullable=False),
        sa.Column("is_active_for_inference", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_version"),
    )
    op.create_index("ix_model_registry_model_version", "model_registry", ["model_version"])
    op.create_index("ix_model_registry_is_active_for_inference", "model_registry", ["is_active_for_inference"])
    op.create_index("ix_model_registry_active_arch", "model_registry", ["is_active_for_inference", "architecture"])


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("analyst_feedback")
    op.drop_table("transaction_explanations")
    op.drop_table("transaction_edges")
    op.drop_table("transactions")
