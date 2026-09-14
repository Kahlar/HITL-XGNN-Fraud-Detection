# Reproducibility Guide & Engineering Runbook

---

## 1. Overview & Reproduction Philosophy

This runbook provides complete, deterministic instructions to reproduce the data pipelines, model training experiments (EXP-01 through EXP-07), evaluation benchmarks, test suites, and Dockerized production services of the **HITL-XGNN Fraud Detection** system.

### Core Principles
- **Deterministic Seed Governance**: All pseudo-random number generators across Python `random`, `numpy`, PyTorch CPU/CUDA, and scikit-learn are locked to global seed `42`.
- **Zero Temporal Data Leakage**: Standard scalers and transformers are fitted exclusively on training timesteps $t \in [1, 34]$. Future time windows ($t \in [35, 49]$) are strictly held out.
- **Immutable Artifacts**: Historical checkpoints in `models/` and evaluation results in `results/` are archived with strict schema validation.
- **Reproducible Containerization**: Multi-stage Docker builds ensure deterministic environment isolation across operating systems.

---

## 2. Hardware & Environment Specifications

### Minimum & Recommended System Requirements

| Specification | Minimum Requirement | Recommended Production / Training |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Ubuntu 22.04 LTS or Windows 11 Pro |
| **CPU** | 4 Cores (x86_64 or Apple Silicon) | 8+ Cores (e.g., AMD Ryzen 7 / Intel Core i7) |
| **RAM** | 8 GB | 16 GB - 32 GB |
| **Disk Space** | 10 GB free space | 25 GB free space (SSD recommended) |
| **GPU** | Not required (CPU verified) | NVIDIA GPU (CUDA 11.8+ / 12.1+) for faster training |
| **Python Version** | Python 3.11, 3.12, or 3.13 | Python 3.11.x |
| **Node.js Runtime** | Node.js v18 LTS or v20 LTS | Node.js v20.x LTS |
| **Docker Engine** | Docker Engine 24.0+ & Compose v2+ | Docker Desktop 4.25+ / Docker CE 24+ |

---

## 3. Repository Structure & Directory Layout

```
HITL-XGNN-Fraud-Detection/
├── configs/
│   └── hitl_config.yaml               # Central experiment, pipeline, and model hyperparameter configuration
├── data/
│   ├── raw/                           # Raw Elliptic CSV dataset (features, edges, classes)
│   └── processed/
│       ├── dataset_summary.json       # Structural validation metadata
│       ├── scaled_features.npy        # Zero-leakage standardized node features
│       └── graphs/                    # 49 PyTorch Geometric graph snapshots (timestep_1.pt .. timestep_49.pt)
├── docker/
│   ├── Dockerfile.backend             # Multi-stage Python 3.11 FastAPI backend image
│   ├── Dockerfile.frontend            # Multi-stage Node 20 / Nginx React frontend image
│   └── docker-compose.yml             # Orchestration for DB, Backend, and Frontend containers
├── models/                            # 22 Model checkpoints (.pt and .joblib)
│   ├── tabular/                       # EXP-01: XGBoost, LightGBM, Random Forest, MLP
│   ├── gnn_architectures/             # EXP-02: GraphSAGE, GAT, GCN
│   ├── feature_ablations/             # EXP-03: Feature subset checkpoints
│   ├── loss_ablations/                # EXP-04: Cross-Entropy and Weighted CE
│   └── hitl_checkpoints/              # EXP-07: Active learning retrained models (uncertainty/random)
├── results/
│   ├── exp01_baselines/               # Tabular benchmark metrics and PR curves
│   ├── exp02_gnn_architectures/       # GNN benchmark metrics and PR curves
│   ├── exp03_feature_ablation/        # Feature ablation comparison logs
│   ├── exp04_loss_functions/          # Focal vs CE vs Weighted CE metrics
│   ├── exp05_explainability/          # GNNExplainer fidelity, sparsity, and latency metrics
│   ├── exp06_temporal_drift/          # Out-of-time per-timestep performance logs
│   └── exp07_hitl/                    # Active learning budget and strategy curves
├── scripts/                           # Phase-by-phase execution scripts (run_phase1.py .. run_phase7.py)
├── src/                               # Modular source code
│   ├── common/                        # Shared logging, config, and utility functions
│   ├── module1_dataset/               # Dataset loading, verification, and EDA
│   ├── module2_preprocessing/         # Feature engineering and TrainOnlyScaler
│   ├── module3_graph_builder/         # PyG graph construction and serialization
│   ├── module4_gnn/                   # Tabular baselines, GNN models, Focal Loss, and training loops
│   ├── module5_explainability/        # GNNExplainer, attention extraction, and fidelity benchmarks
│   ├── module5_temporal_drift/        # Out-of-time temporal drift and macro-shock evaluator
│   ├── module6_api/                   # FastAPI backend service, schemas, routers, and React frontend
│   ├── module7_hitl/                  # Active learning query strategies and retraining pipeline
│   └── module8_database/              # PostgreSQL SQLAlchemy ORM, connection pool, repositories
├── tests/
│   └── unit/                          # 76 passing pytest unit and integration tests
├── alembic/                           # Database migration scripts
├── alembic.ini                        # Alembic configuration
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Production Python dependencies
├── requirements-dev.txt               # Development and test dependencies
├── README.md                          # Repository overview and primary quickstart
├── RESEARCH.md                        # Publication-grade research compendium
└── REPRODUCIBILITY.md                 # This reproduction runbook
```

---

## 4. Complete End-to-End Pipeline Execution Guide

To reproduce the entire scientific lifecycle from raw CSV ingestion through to the production database and HITL active learning, execute the sequential phase runner scripts in order:

```bash
# 1. Activate your Python virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Phase 1: Dataset Verification & Ingestion
python scripts/run_phase1.py

# 3. Phase 2: Feature Engineering & Zero-Leakage Preprocessing
python scripts/run_phase2.py

# 4. Phase 3: PyTorch Geometric Graph Serialization (49 Timesteps)
python scripts/run_phase3.py

# 5. Phase 4: Tabular Baselines (EXP-01) & GNN Architectures (EXP-02)
python scripts/run_phase4.py

# 6. Phase 5: GNNExplainer XAI Benchmarks (EXP-05) & Temporal Shock (EXP-06)
python scripts/run_phase5.py

# 7. Phase 6: Full Active Learning Benchmark (EXP-07)
python scripts/run_phase6.py

# 8. Phase 7: Verification of Experiment Artifacts & Integrity
python scripts/run_phase7.py
```

---

## 5. Step-by-Step Individual Experiment Reproduction

### EXP-01: Tabular Baselines
Trains and evaluates XGBoost, LightGBM, Random Forest, and 3-Layer MLP on the 165-dimensional node features.
```bash
python -c "
from src.module4_gnn.baselines import run_all_baselines
from src.common.config import load_config
cfg = load_config('configs/hitl_config.yaml')
results = run_all_baselines(cfg)
"
```
- **Primary Outputs**: `models/tabular/`, `results/exp01_baselines/tabular_baselines_summary.json`
- **Expected Benchmark**: XGBoost Test F1 = `0.7223`, PR-AUC = `0.6712` ($\tau^* = 0.9014$).

### EXP-02: GNN Architecture Comparison
Trains GraphSAGE, GAT (4-Head), and GCN with Focal Loss ($\alpha=0.25, \gamma=2.0$) on serialized PyG graphs ($t \in [1, 34]$).
```bash
python -c "
from src.module4_gnn.train import train_and_evaluate_all_gnns
from src.common.config import load_config
cfg = load_config('configs/hitl_config.yaml')
results = train_and_evaluate_all_gnns(cfg)
"
```
- **Primary Outputs**: `models/gnn_architectures/`, `results/exp02_gnn_architectures/gnn_architectures_summary.json`
- **Expected Benchmark**: GraphSAGE Test F1 = `0.5175`, Val PR-AUC = `0.7966` ($\tau^* = 0.5517$).

### EXP-03: Feature Ablation Study
Evaluates GraphSAGE performance across 4 feature subsets (`original_all`, `combined`, `original_local`, `engineered`).
```bash
python -c "
from src.module4_gnn.ablation import run_feature_ablation
from src.common.config import load_config
cfg = load_config('configs/hitl_config.yaml')
results = run_feature_ablation(cfg)
"
```
- **Primary Outputs**: `models/feature_ablations/`, `results/exp03_feature_ablation/feature_ablation_summary.json`
- **Expected Benchmark**: `original_all` (165 feats) Test F1 = `0.5139` vs `combined` (170 feats) Test F1 = `0.4197`.

### EXP-04: Loss Function Optimization
Compares Focal Loss ($\alpha=0.25, \gamma=2.0$), Standard Binary Cross-Entropy, and Weighted Cross-Entropy ($w=7.63$).
```bash
python scripts/rerun_exp04.py
```
- **Primary Outputs**: `models/loss_ablations/`, `results/exp04_loss_functions/loss_comparison_summary.json`
- **Expected Benchmark**: Focal Loss Val PR-AUC = `0.7966`, Test F1 = `0.5175` vs Standard CE Test F1 = `0.4888`.

### EXP-05: Subgraph Explainability & Fidelity
Computes $\text{Fidelity}^+$, $\text{Fidelity}^-$, edge/feature sparsity, and execution latency for GNNExplainer and GAT Attention.
```bash
python -c "
from src.module5_explainability.benchmark import run_explainability_benchmark
from src.common.config import load_config
cfg = load_config('configs/hitl_config.yaml')
results = run_explainability_benchmark(cfg)
"
```
- **Primary Outputs**: `results/exp05_explainability/xai_benchmark_summary.json`
- **Expected Benchmark**: GNNExplainer $\text{Fidelity}^+ = 0.1470$, $\text{Fidelity}^- = 0.0069$, Sparsity = `0.6842`, Mean Latency = `292.25 ms`.

### EXP-06: Temporal Drift & Darknet Shock Evaluation
Evaluates out-of-time degradation across test timesteps $t \in [40, 49]$, highlighting the darknet shutdown shock ($t=43..46$).
```bash
python -c "
from src.module5_temporal_drift.drift_evaluator import evaluate_temporal_drift
from src.common.config import load_config
cfg = load_config('configs/hitl_config.yaml')
results = evaluate_temporal_drift(cfg)
"
```
- **Primary Outputs**: `results/exp06_temporal_drift/temporal_drift_summary.json`
- **Expected Benchmark**: Pre-Shock ($t=40..42$) Mean F1 = `0.6381` $\to$ Shock ($t=43..46$) Mean F1 = `0.0000` (Prevalence `0.99%`).

### EXP-07: Human-in-the-Loop Active Learning Benchmark
Simulates active feedback queries from validation candidates across $5\%$, $10\%$, and $20\%$ budgets using Uncertainty vs Random sampling.
```bash
python scripts/run_exp07_hitl.py
```
- **Primary Outputs**: `models/hitl_checkpoints/`, `results/exp07_hitl/hitl_summary.json`
- **Expected Benchmark**: Uncertainty 20% Budget Test F1 = `0.5030` vs Random 20% Budget Test F1 = `0.4401` ($+6.29\%$ relative gain).

---

## 6. Determinism & Random Seed Governance

Deterministic execution is enforced globally across all modules in `src/common/utils.py`:

```python
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """Sets global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

All training scripts initialize `set_global_seed(cfg.get("random_seed", 42))` prior to dataset partitioning and parameter initialization.

---

## 7. Data Artifacts & Checkpoint Reference

### Graph Snapshot Artifacts (`data/processed/graphs/`)
- Serialized as 49 individual PyTorch Geometric `Data` objects (`timestep_1.pt` to `timestep_49.pt`).
- Each artifact encapsulates:
  - `data.x`: Standardized node features (`torch.FloatTensor` of shape `[N_t, 165]`).
  - `data.edge_index`: Directed payment edges (`torch.LongTensor` of shape `[2, E_t]`).
  - `data.y`: Ground-truth labels (`0` = Licit, `1` = Illicit, `-1` = Unknown).
  - `data.tx_id`: Canonical Elliptic transaction ID strings (`List[str]`).
  - `data.train_mask`, `data.val_mask`, `data.test_mask`: Boolean split tensors.

### Checkpoint Inventory Summary
| Checkpoint Path | Architecture | Experiment | Key Metric |
| :--- | :--- | :--- | :--- |
| `models/tabular/xgboost_baseline.joblib` | XGBoost (Tree) | EXP-01 | Test F1 = 0.7223 |
| `models/gnn_architectures/graphsage_focal.pt` | GraphSAGE | EXP-02 | Test F1 = 0.5175 |
| `models/hitl_checkpoints/graphsage_hitl_uncertainty_20.pt` | GraphSAGE (HITL) | EXP-07 | Test F1 = 0.5030 (**Production Active**) |

---

## 8. Unit & Integration Testing Runbook

### Backend Test Suite (76 Tests)
Run all backend unit, integration, and database tests:
```bash
# Execute complete backend pytest suite
pytest tests/unit/ -v
```

```
============================== test session starts ==============================
platform win32 -- Python 3.14.7, pytest-9.1.1
collected 76 items

tests/unit/test_canonical_tx_ids.py ......                                [  7%]
tests/unit/test_docker_config.py .....                                    [ 14%]
tests/unit/test_gnn_models.py .........                                   [ 26%]
tests/unit/test_module1_dataset.py ......                                 [ 34%]
tests/unit/test_module2_preprocessing.py ......                           [ 42%]
tests/unit/test_module3_graph_builder.py .......                          [ 51%]
tests/unit/test_module4_baselines.py .......                              [ 60%]
tests/unit/test_module5_explainability.py ......                          [ 68%]
tests/unit/test_module5_temporal_drift.py .....                           [ 75%]
tests/unit/test_module6_api.py .........                                  [ 86%]
tests/unit/test_module7_hitl.py .......                                   [ 96%]
tests/unit/test_module8_database.py ...                                   [100%]

============================== 76 passed in 14.23s ==============================
```

### Frontend Test Suite (9 Tests)
Run all frontend React component and utility tests:
```bash
cd src/module6_api/frontend
npm test -- --run
```

```
 ✓ src/test/App.test.tsx (9 tests) 312ms
   ✓ renders dashboard navigation without crashing
   ✓ loads transaction explorer view
   ✓ renders graph canvas container
   ✓ displays XAI explanation modal
   ✓ handles HITL review submission
   ✓ formats transaction amounts correctly
   ✓ displays risk badge color coding
   ✓ filters transactions by risk level
   ✓ renders temporal drift metrics table

 Test Files  1 passed (1)
      Tests  9 passed (9)
```

---

## 9. Database Migration & Seeding Runbook

### 1. Execute Alembic Migrations
Ensure the database schema is up-to-date with all indexes and constraints:
```bash
# Apply migrations to head
alembic upgrade head
```

### 2. Verify Schema Tables in PostgreSQL
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- Expected Output:
-- transactions
-- transaction_edges
-- transaction_explanations
-- analyst_feedback
-- model_registry
-- alembic_version
```

---

## 10. Docker Multi-Container Deployment Runbook

### 1. Build and Launch Multi-Container Application
Launch PostgreSQL, the FastAPI Backend, and the React Frontend simultaneously:
```bash
# Start all containers in detached mode with fresh build
docker compose -f docker/docker-compose.yml up --build -d
```

### 2. Verify Container Health and Status
```bash
docker compose -f docker/docker-compose.yml ps
```

Expected output:
```
NAME                 IMAGE                         STATUS                    PORTS
hitl-fraud-db        postgres:16-alpine            Up (healthy)              0.0.0.0:5432->5432/tcp
hitl-fraud-backend   hitl-fraud-backend:latest     Up (healthy)              0.0.0.0:8000->8000/tcp
hitl-fraud-frontend  hitl-fraud-frontend:latest    Up (healthy)              0.0.0.0:3000->80/tcp
```

### 3. Teardown
```bash
# Stop containers and preserve database volume
docker compose -f docker/docker-compose.yml down

# To also remove database volume (CAUTION):
# docker compose -f docker/docker-compose.yml down -v
```

---

## 11. API Verification & Curl Cheatsheet

Once the backend is running at `http://localhost:8000`:

```bash
# 1. System Health Check
curl -s http://localhost:8000/api/v1/health | jq .

# 2. Query Transactions (Paginated, Timestep 40)
curl -s "http://localhost:8000/api/v1/transactions?page=1&page_size=5&timestep=40" | jq .

# 3. Retrieve Single Transaction Detail
curl -s "http://localhost:8000/api/v1/transactions/230425980" | jq .

# 4. Extract 2-Hop Local Payment Subgraph
curl -s "http://localhost:8000/api/v1/graph/230425980/subgraph?hops=2" | jq .

# 5. Generate GNNExplainer XAI Report
curl -s "http://localhost:8000/api/v1/explain/230425980" | jq .

# 6. Query HITL Triage Priority Queue
curl -s "http://localhost:8000/api/v1/hitl/queue?page=1&page_size=5" | jq .

# 7. Submit Analyst Review Feedback
curl -s -X POST "http://localhost:8000/api/v1/hitl/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "tx_id": "230425980",
       "analyst_id": "analyst_alice",
       "analyst_label": 1,
       "confidence": 0.95,
       "notes": "Verified fan-out pattern matching mixing service deposit"
     }' | jq .

# 8. Retrieve Analytics & Temporal Metrics
curl -s "http://localhost:8000/api/v1/analytics/metrics" | jq .
```

---

## 12. Frontend Development & Build Runbook

```bash
# Navigate to frontend directory
cd src/module6_api/frontend

# Install dependencies
npm install

# Start Vite live development server (http://localhost:3000)
npm run dev

# Run TypeScript typecheck and production build
npm run build

# Run Vitest unit tests
npm test -- --run
```

---

## 13. Troubleshooting & Common Issues

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| **Port Conflict (8000, 3000, or 5432)** | Existing process or container occupies port | Identify process via `netstat -ano \| findstr :8000` (Win) or `lsof -i :8000` (Linux) and terminate, or change host port mapping in `docker/docker-compose.yml`. |
| **PyTorch Geometric Wheel Missing** | Installing PyG without compatible torch wheels | Run `pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`. |
| **Database Connection Timeout** | PostgreSQL container starting slower than backend | Docker healthcheck handles this automatically; if running locally, verify `DATABASE_URL` matches PostgreSQL port and credentials. |
| **Docker Out of Memory** | Docker daemon memory limit below 4GB | Increase allocated RAM to $\ge 6\text{ GB}$ in Docker Desktop settings $\to$ Resources $\to$ Memory. |
| **Graph Artifact Not Found** | Pipeline Phase 3 not executed | Execute `python scripts/run_phase3.py` to generate `data/processed/graphs/timestep_*.pt`. |

---

## 14. Canonical Results Checksum & Verification Matrix

| Experiment ID | Execution Script / Command | Primary Result Artifact | Canonical Benchmark Metric |
| :--- | :--- | :--- | :--- |
| **EXP-01** | `python scripts/run_phase4.py` | `results/exp01_baselines/tabular_baselines_summary.json` | XGBoost Test F1 = **0.7223** |
| **EXP-02** | `python scripts/run_phase4.py` | `results/exp02_gnn_architectures/gnn_architectures_summary.json` | GraphSAGE Test F1 = **0.5175** |
| **EXP-03** | `python scripts/run_phase4.py` | `results/exp03_feature_ablation/feature_ablation_summary.json` | `original_all` Test F1 = **0.5139** |
| **EXP-04** | `python scripts/rerun_exp04.py` | `results/exp04_loss_functions/loss_comparison_summary.json` | Focal Loss Test F1 = **0.5175** |
| **EXP-05** | `python scripts/run_phase5.py` | `results/exp05_explainability/xai_benchmark_summary.json` | GNNExplainer Fidelity$^+$ = **0.1470** |
| **EXP-06** | `python scripts/run_phase5.py` | `results/exp06_temporal_drift/temporal_drift_summary.json` | Pre-Shock Mean F1 = **0.6381** |
| **EXP-07** | `python scripts/run_exp07_hitl.py` | `results/exp07_hitl/hitl_summary.json` | Uncertainty 20% F1 = **0.5030** |
