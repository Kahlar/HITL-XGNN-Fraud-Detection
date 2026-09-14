"""Phase 1 Execution Script: Dataset validation, preprocessing, and temporal splitting."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import default_config
from src.common.logger import get_logger
from src.common.schemas import PreprocessingMetadata
from src.module1_dataset.loader import EllipticDataLoader
from src.module1_dataset.validator import DatasetValidator
from src.module2_preprocessing.cleaner import LabelCleaner
from src.module2_preprocessing.feature_engineering import GraphFeatureEngineer
from src.module2_preprocessing.scaler import TrainOnlyScaler
from src.module2_preprocessing.splitter import TemporalSplitter

logger = get_logger("Phase1.Runner")


def main() -> int:
    """Executes the complete Phase 1 pipeline and verifies all outputs."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 1: ENVIRONMENT + DATASET VALIDATION + PREPROCESSING")
    logger.info("=" * 80)

    raw_dir = default_config.paths.raw_data_dir
    processed_dir = default_config.paths.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Module 1 — Dataset Ingestion & Validation
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 1: VALIDATING RAW DATASET (MODULE 1) ---")
    validator = DatasetValidator(raw_data_dir=raw_dir, output_dir=processed_dir)
    validation_report = validator.validate_and_generate_report()

    if not validation_report.validation_passed:
        logger.error("Dataset validation FAILED. Exiting with error code 1.")
        return 1

    logger.info("Dataset validation PASSED successfully.")

    # -------------------------------------------------------------------------
    # Step 2: Module 2 — Label Cleaning
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 2: CLEANING LABELS (MODULE 2) ---")
    loader = EllipticDataLoader(raw_data_dir=raw_dir)
    df_classes = loader.load_classes()
    cleaner = LabelCleaner()
    df_labels = cleaner.clean_labels(df_classes)

    # -------------------------------------------------------------------------
    # Step 3: Module 2 — Temporal Splitting & Split Metadata
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 3: TEMPORAL SPLITTING (MODULE 2) ---")
    df_timesteps = loader.load_transaction_timesteps()
    
    # Merge timesteps and labels
    df_merged = pd.merge(df_timesteps, df_labels, on="txId", how="inner")
    if len(df_merged) != len(df_classes):
        logger.error(f"Mismatch after merging classes and timesteps: {len(df_merged)} != {len(df_classes)}")
        return 1

    splitter = TemporalSplitter(output_dir=processed_dir)
    df_split, split_metadata = splitter.split_and_generate_metadata(df_merged)

    # -------------------------------------------------------------------------
    # Step 4: Module 2 — Feature Engineering (Graph Topological Features)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 4: COMPUTING GRAPH TOPOLOGICAL FEATURES (MODULE 2) ---")
    df_edges = loader.load_edges()
    engineer = GraphFeatureEngineer()
    df_engineered = engineer.compute_graph_features(df_split, df_edges)

    # -------------------------------------------------------------------------
    # Step 5: Module 2 — Train-Only Feature Scaling
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 5: FITTING SCALER STRICTLY ON TRAIN SPLIT (MODULE 2) ---")
    df_features = loader.load_features(use_float32=True)
    
    # Get combined features matrix (165 original + 5 engineered = 170)
    mat_combined, combined_cols = engineer.get_feature_matrix(
        df_features=df_features,
        df_engineered=df_engineered,
        config="combined"
    )

    # Identify train split indices
    train_mask = (df_split["split"] == "train").to_numpy()
    val_mask = (df_split["split"] == "val").to_numpy()
    test_mask = (df_split["split"] == "test").to_numpy()

    X_train = mat_combined[train_mask]
    X_val = mat_combined[val_mask]
    X_test = mat_combined[test_mask]

    scaler = TrainOnlyScaler(scaler_type="robust", output_dir=processed_dir)
    scaler.fit(X_train, train_timesteps=default_config.splits.train_range)
    
    # Verify transform works on val and test without errors
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    scaler_save_path = scaler.save("feature_scaler.joblib")

    # -------------------------------------------------------------------------
    # Step 6: Save Preprocessing Metadata & Artifacts Summary
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 6: SAVING PREPROCESSING METADATA ---")
    meta = PreprocessingMetadata(
        num_total_transactions=len(df_merged),
        label_mapping={"1": 1, "2": 0, "unknown": -1},
        original_all_feature_dim=165,
        original_local_feature_dim=93,
        engineered_feature_dim=len(combined_cols) - 165,
        combined_feature_dim=len(combined_cols),
        scaler_type="robust",
        scaler_fit_timesteps=default_config.splits.train_range,
        saved_artifacts={
            "validation_report_json": str(processed_dir / "dataset_validation_report.json"),
            "validation_report_txt": str(processed_dir / "dataset_validation_report.txt"),
            "split_metadata_json": str(processed_dir / "split_metadata.json"),
            "feature_scaler_joblib": str(scaler_save_path),
            "preprocessing_metadata_json": str(processed_dir / "preprocessing_metadata.json"),
        }
    )

    meta_path = processed_dir / "preprocessing_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.model_dump_json(indent=2))
    logger.info(f"Saved preprocessing metadata to {meta_path}")

    elapsed = datetime.now() - start_time

    # -------------------------------------------------------------------------
    # Final Verification Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1 EXECUTION AND VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Status                  : SUCCESS (All validation checks passed)")
    print(f"Execution Duration      : {elapsed.total_seconds():.2f} seconds")
    print(f"Total Transactions      : {len(df_merged):,}")
    print(f"Total Graph Edges       : {len(df_edges):,}")
    print(f"Unique Timesteps        : {split_metadata.train.timesteps[0]}..{split_metadata.test.timesteps[-1]} (Total: {len(split_metadata.per_timestep_stats)})")
    print(f"Intra-Timestep Edges    : {validation_report.edge_statistics.pct_intra_timestep:.2f}% ({validation_report.edge_statistics.intra_timestep_edges:,})")
    print(f"Self-Loops              : {validation_report.edge_statistics.self_loops}")
    print(f"Missing / NaN Values    : {validation_report.missing_or_nan_values_found}")
    print("-" * 80)
    print(f"Train Partition (t=1..34) : {split_metadata.train.total_nodes:,} nodes | {split_metadata.train.labeled_nodes:,} labeled ({split_metadata.train.illicit_count:,} illicit, {split_metadata.train.licit_count:,} licit) | {split_metadata.train.pct_illicit_in_labeled:.2f}% illicit")
    print(f"Val Partition (t=35..39)  : {split_metadata.validation.total_nodes:,} nodes | {split_metadata.validation.labeled_nodes:,} labeled ({split_metadata.validation.illicit_count:,} illicit, {split_metadata.validation.licit_count:,} licit) | {split_metadata.validation.pct_illicit_in_labeled:.2f}% illicit")
    print(f"Test Partition (t=40..49) : {split_metadata.test.total_nodes:,} nodes | {split_metadata.test.labeled_nodes:,} labeled ({split_metadata.test.illicit_count:,} illicit, {split_metadata.test.licit_count:,} licit) | {split_metadata.test.pct_illicit_in_labeled:.2f}% illicit")
    print("-" * 80)
    print(f"Feature Dimensions      : Original Local = 93 | Original All = 165 | Engineered = 5 | Combined = 170")
    print(f"Scaler Training Policy  : Fitted strictly on train timesteps t=1..34 (Zero data leakage)")
    print(f"Generated Artifacts     :")
    for k, v in meta.saved_artifacts.items():
        print(f"  - {k:<28}: {v}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
