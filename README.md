# Human-in-the-Loop Explainable Graph Neural Network (HITL-XGNN) for Financial Fraud Detection

A research-grade, paper-ready financial fraud detection system combining the Elliptic Bitcoin transaction graph, Graph Neural Networks (GNNs), Explainable AI (XAI), human reviewer workflows, and an active learning feedback loop.

---

## 1. Project Overview & Research Objectives

The **HITL-XGNN** framework addresses the severe class imbalance, structural relational dependencies, post-hoc interpretability requirements, and out-of-time temporal distribution shift inherent in cryptocurrency anti-money laundering (AML) and financial transaction monitoring.

The core research objectives are:
1. **Relational Representation Learning**: Quantitatively evaluate Graph Neural Networks against state-of-the-art tabular gradient boosting baselines under strict temporal splits.
2. **Post-Hoc Explainability & Faithfulness**: Measure the empirical fidelity ($\text{Fidelity}^+$, $\text{Fidelity}^-$), sparsity, and attribution stability of `GNNExplainer` against multi-head attention and perturbation baselines.
3. **Out-of-Time Temporal Robustness**: Quantify performance degradation during non-stationary macro shocks (the darknet marketplace shutdown window at $t=43..46$).
4. **Human-in-the-Loop Sample Efficiency**: Evaluate whether uncertainty-guided active learning querying improves model retraining sample efficiency over random and heuristic feedback selection.

---

## 2. Canonical Eight-Module System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         HITL-XGNN SYSTEM ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────────┘
  [Raw Elliptic CSVs] ──► Module 1: Ingestion & Validation
                                 │
                                 ▼
                          Module 2: Temporal Splitting & Preprocessing
                          (Train: t=1..34 | Val: t=35..39 | Test: t=40..49)
                                 │
                                 ▼
                          Module 3: PyG Temporal Graph Builder
                          (49 Disjoint Subgraphs G_1..G_49, 170 Features)
                                 │
                                 ▼
                          Module 4: GNN & Tabular Model Training
                          (GraphSAGE + Focal Loss: alpha=0.25, gamma=2.0)
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     Module 5: XAI        Module 7: HITL        Module 8: PostgreSQL
    (GNNExplainer &       (Triage Router &        (Relational Feedback &
    Fidelity Benchmark)   Active Learning)        Model Registry Store)
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                          Module 6: FastAPI Async API + React 18 Analyst UI
```

1. **Module 1 — Dataset**: Ingestion, validation, and schema auditing of raw transaction, edge, and class files. *(Phase 1)*
2. **Module 2 — Preprocessing & Feature Engineering**: Label mapping, temporal disjoint splitting, graph topological feature engineering (5 features), and train-only feature scaling. *(Phase 1)*
3. **Module 3 — Graph Builder**: Construction of PyTorch Geometric graphs per discrete time step ($G_1 \dots G_{49}$), $k$-hop computational graph extractors, and topological statistics. *(Phase 2)*
4. **Module 4 — GNN / Model Training & Inference**: Tabular baselines (XGBoost, LightGBM, Random Forest, MLP) and Graph Neural Networks (GCN, GraphSAGE, GAT) optimized with Focal Loss. *(Phases 3 & 4)*
5. **Module 5 — Explainability**: Post-hoc attribution using `GNNExplainer` with quantitative $\text{Fidelity}^+$ and $\text{Fidelity}^-$ validation against GAT attention and Random baselines. *(Phase 6)*
6. **Module 6 — FastAPI + React Application Layer**: High-performance async backend and interactive 2D graph visualizer canvas. *(Backend in Phase 9, React 18 Frontend in Phase 10)*
7. **Module 7 — Human Analyst / HITL / Active Learning**: Uncertainty-guided triage routing, reviewer audit logging, and active learning retraining (EXP-07). *(Phase 7)*
8. **Module 8 — PostgreSQL Feedback & Model Registry**: Relational persistence for predictions, risk levels, explanations, reviewer verdicts, and versioned checkpoints. *(Phase 8)*

---

## 3. Verified Dataset Facts (Observed Programmatically)

- **Raw Dataset Location**: `data/raw/elliptic_bitcoin_dataset/`
- **Total Transactions**: 203,769 unique nodes across all files (0 duplicates, 0 missing values, 0 non-finite values).
- **Total Payment Edges**: 234,355 directed edges (0 self-loops).
- **Intra-Timestep Edges**: Exactly 234,355 (**100.00%**). Edges exist strictly within the same 2-week time step. There are 0 cross-timestep edges.
- **Total Timesteps**: 49 discrete time steps.
- **Class Distribution**:
  - `unknown` (unlabeled): 157,205 (77.15%)
  - `licit` (Class 2): 42,019 (20.62% total, 90.24% of labeled)
  - `illicit` (Class 1): 4,545 (2.23% total, 9.76% of labeled)
  - Labeled Class Imbalance Ratio: $9.25 : 1$ (licit to illicit).

---

## 4. Canonical Temporal Splitting Protocol

To prevent temporal lookahead leakage and evaluate true out-of-time generalizability, transactions are partitioned strictly by discrete timestep:

| Split Partition | Timestep Range | Total Nodes | Labeled Nodes | Illicit Cases | Licit Cases | Imbalance Ratio | Partition Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train Split** | $t \in [1, 34]$ | 136,875 | 29,894 | 3,462 | 26,432 | 7.63 : 1 | Feature scaling fit, base model training |
| **Validation Split** | $t \in [35, 39]$ | 24,064 | 5,486 | 447 | 5,039 | 11.27 : 1 | Early stopping, decision threshold tuning ($\tau^*$), HITL feedback candidate pool |
| **Test Split** | $t \in [40, 49]$ | 42,830 | 11,184 | 636 | 10,548 | 16.59 : 1 | **Untouched out-of-time evaluation**, temporal shock benchmark |
| **Total** | $t \in [1, 49]$ | **203,769** | **46,564** | **4,545** | **42,019** | 9.25 : 1 | Complete Elliptic Graph |

---

## 5. Master Empirical Benchmark Results

### A. Model Performance Comparison (Evaluated on Untouched Test Partition $t \in [40, 49]$)

All models evaluated with their respective decision thresholds $\tau^*$ calibrated strictly on validation data ($t=35..39$):

| Experiment | Model Architecture | Model Family | Feature Set | Calibrated Threshold $\tau^*$ | Test F1 (Illicit) | Test PR-AUC | Test Precision | Test Recall | Test ROC-AUC | Parameters |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | **XGBoost** | Tabular Gradient Boosting | `original_all` (165) | 0.9014 | **0.7223** | **0.6712** | 0.9608 | 0.5786 | 0.8828 | — |
| **EXP-01** | **LightGBM** | Tabular Gradient Boosting | `original_all` (165) | 0.8915 | 0.7179 | 0.6692 | 0.9630 | 0.5723 | 0.8908 | — |
| **EXP-01** | **Random Forest** | Tabular Bagging Ensemble | `original_all` (165) | 0.6305 | 0.7061 | 0.6662 | 0.9215 | 0.5723 | 0.8912 | — |
| **EXP-01** | **MLP (3-Layer)** | Deep Neural Network | `original_all` (165) | 0.9408 | 0.5714 | 0.5321 | 0.8034 | 0.4434 | 0.8812 | 23,298 |
| **EXP-02 / 04** | **GraphSAGE** | Graph Neural Network | `original_all` (165) | 0.5517 | **0.5175** | **0.4499** | 0.7456 | 0.3962 | 0.8289 | 43,138 |
| **EXP-02** | **GAT (4-Head)** | Graph Neural Network | `original_all` (165) | 0.5468 | 0.4097 | 0.3183 | 0.6238 | 0.3050 | 0.7911 | 22,022 |
| **EXP-02** | **GCN** | Graph Neural Network | `original_all` (165) | 0.5665 | 0.3067 | 0.2087 | 0.4386 | 0.2358 | 0.7137 | 21,762 |

### B. Feature Configuration Ablation (EXP-03 on GraphSAGE)

| Configuration | Feature Count | Features Included | Validation F1 | Test F1 | Test PR-AUC | Test Precision | Test Recall |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`original_all`** | 165 | 93 local + 72 aggregated 1-hop neighbor statistics | 0.7294 | **0.5139** | **0.4615** | 0.7418 | 0.3931 |
| **`combined`** | 170 | 165 original + 5 engineered graph topological features | 0.6317 | 0.4197 | 0.4201 | 0.5452 | 0.3412 |
| **`original_local`** | 93 | 93 local transaction features only | 0.7171 | 0.3780 | 0.3800 | 0.7900 | 0.2484 |
| **`engineered`** | 5 | In/out degree, total degree, in/out ratio, flow balance | 0.2872 | 0.1804 | 0.1291 | 0.1311 | 0.2893 |

### C. Loss Function & Imbalance Benchmark (EXP-04 on GraphSAGE 165 Features)

| Loss Function | Loss Formulation & Hyperparameters | Validation PR-AUC | Validation F1 | Test F1 | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Focal Loss** | $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$, $\alpha=0.25, \gamma=2.0$ | **0.7966** | 0.7299 | **0.5175** | **0.4499** |
| **Standard Cross Entropy** | $\text{CE}(p) = -[y \log(p) + (1-y)\log(1-p)]$ | 0.7958 | **0.7339** | 0.4888 | 0.4373 |
| **Weighted Cross Entropy** | $w_{\text{illicit}} = \frac{N_{\text{licit}}}{N_{\text{illicit}}} = 7.63$ | 0.7735 | 0.7108 | 0.4571 | 0.3975 |

---

## 6. Explainability & Fidelity Findings (EXP-05)

Evaluated across 48 stratified test transactions across TP, FN, FP, and TN quadrants and all three temporal regimes:

| Attribution Method | Mean $\text{Fidelity}^+$ (Necessity) $\uparrow$ | Mean $\text{Fidelity}^-$ (Sufficiency) $\downarrow$ | Mean Edge Sparsity | Mean Feature Sparsity | Mean Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNNExplainer** | **0.1470** | **0.0069** | 0.6842 | 0.7333 | 292.25 ms | 378.99 ms |
| **GAT Internal Attention** | 0.1069 | 0.0141 | 0.5000 | — | **0.42 ms** | 0.65 ms |
| **Random Baseline** | 0.0134 | 0.1541 | 0.6842 | 0.7333 | 0.12 ms | 0.18 ms |

- **Necessity ($\text{Fidelity}^+$)**: Masking GNNExplainer top features causes a 14.70% probability drop (vs 1.34% for random masks), confirming that identified subgraphs carry the core prediction signal.
- **Sufficiency ($\text{Fidelity}^-$)**: Retaining only GNNExplainer top features preserves prediction confidence with minimal degradation (0.69% error vs 15.41% for random).

---

## 7. Out-of-Time Temporal Drift & Macro Shock (EXP-06)

Evaluated on the frozen GraphSAGE model ($\tau^* = 0.5517$) across the 10 discrete test timesteps ($t \in [40, 49]$):

```
F1 Score Progression Across Test Timesteps:
1.00 ┤
0.80 ┤                ┌──┐ [t=42: 0.7643]
0.60 ┤   ┌──┐ [0.595] │  │
0.40 ┤   │  │  ┌──┐   │  │
0.20 ┤   │  │  │  │   │  │
0.00 ┼───┴──┴──┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴──
       t=40  t=41   t=42   t=43  t=44  t=45  t=46  t=47  t=48  t=49
       [  Pre-Shock   ]   [    Darknet Shock    ] [    Post-Shock   ]
```

- **Pre-Shock Regime ($t \in [40, 42]$)**: Mean F1 = **0.6381**, Mean PR-AUC = **0.6508**, Mean Illicit Prevalence = **10.20%**.
- **Darknet Shutdown Shock ($t \in [43, 46]$)**: Mean F1 = **0.0000**, Mean PR-AUC = **0.0465**, Illicit Prevalence collapses to **0.99%** (lowest: 0.28% at $t=46$).
- **Post-Shock Regime ($t \in [47, 49]$)**: Mean F1 = **0.0000**, Mean PR-AUC = **0.1353**, Illicit Prevalence rebounds to **7.33%** (peak: 11.76% at $t=49$), but static models fail to adapt without feedback.

---

## 8. Human-in-the-Loop Active Learning (EXP-07)

Simulated feedback was queried strictly from validation timesteps $t \in [35, 39]$ (zero test set leakage) and evaluated on the untouched test partition ($t \in [40, 49]$):

| Feedback Strategy | Query Selection Formula | 0% Budget (Base) | 5% Budget (~274 tx) | 10% Budget (~548 tx) | 20% Budget (~1,097 tx) | Gain vs Random @ 20% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Uncertainty** | $U(x) = 1 - 2|P(x) - 0.5|$ | 0.5175 | **0.4553** | 0.4707 | **0.5030** | **+6.29% F1** |
| **Entropy** | $H(P) = -P\log_2 P - (1-P)\log_2(1-P)$ | 0.5175 | 0.4320 | 0.4738 | 0.4862 | +4.61% F1 |
| **High Risk** | $\text{Score}(x) = P(\text{fraud} \mid x)$ | 0.5175 | 0.4338 | 0.4674 | 0.4949 | +5.48% F1 |
| **Combined Active** | $0.5 \cdot U(x) + 0.5 \cdot P(x)$ | 0.5175 | 0.4501 | **0.4751** | 0.4903 | +5.02% F1 |
| **Random Sampling** | Uniform random selection | 0.5175 | 0.4355 | 0.4591 | 0.4401 | Baseline (0.00%) |

> [!IMPORTANT]
> **Scientific Interpretation**: At a 20% feedback budget, uncertainty-based active querying achieved a **6.29% F1 improvement over random sampling** (0.5030 vs 0.4401), demonstrating greater sample efficiency. However, fine-tuning solely on validation feedback does not establish an overall gain over the frozen base GraphSAGE model (0.5175), demonstrating that human feedback must be paired with temporal adaptation strategies to overcome macro distribution shifts.

---

## 9. Model Checkpoint Inventory

All 22 serialized model artifacts are stored in `models/`:

- **Tabular Baselines (`models/baselines/`)**:
  - `random_forest.joblib`, `xgboost.json`, `lightgbm.txt`, `mlp.pt`
- **Graph Neural Networks (`models/gnn/`)**:
  - `gcn_best.pt`, `graphsage_best.pt` (Base Model), `gat_best.pt`
- **Retrained HITL Checkpoints (`models/gnn/hitl/`)**:
  - `graphsage_hitl_uncertainty_05.pt`, `10.pt`, `20.pt` (Active Production Checkpoint)
  - `graphsage_hitl_high_risk_05.pt`, `10.pt`, `20.pt`
  - `graphsage_hitl_entropy_05.pt`, `10.pt`, `20.pt`
  - `graphsage_hitl_combined_active_05.pt`, `10.pt`, `20.pt`
  - `graphsage_hitl_random_05.pt`, `10.pt`, `20.pt`

---

## 10. Pipeline Reproduction Commands

All empirical experiments can be reproduced sequentially from the repository root:

```bash
# Phase 1: Ingest, validate raw dataset, engineer features, fit train-only scaler
python scripts/run_phase1.py

# Phase 2: Construct and serialize 49 PyG temporal subgraphs
python scripts/run_phase2.py

# Phase 3: Train and evaluate EXP-01 tabular baselines (RF, XGBoost, LightGBM, MLP)
python scripts/run_phase3.py

# Phase 4: Train and benchmark EXP-02 (GNNs), EXP-03 (Features), and EXP-04 (Losses)
python scripts/run_phase4.py

# Phase 5: Evaluate EXP-06 out-of-time temporal drift & darknet shock
python scripts/run_phase5.py

# Phase 6: Execute EXP-05 GNNExplainer fidelity and attribution benchmark
python scripts/run_phase6.py

# Phase 7: Run EXP-07 active learning retraining and sample efficiency benchmark
python scripts/run_phase7.py
```

---

## 11. Multi-Stage Docker Microservice Deployment

The complete application is containerized into a multi-stage, 3-tier microservice architecture:

```bash
# 1. Build and start all 3 containerized services
docker compose -f docker/docker-compose.yml -p hitl-xgnn-fraud-detection up -d --build

# 2. Check health status
docker compose -f docker/docker-compose.yml -p hitl-xgnn-fraud-detection ps

# 3. View logs
docker compose -f docker/docker-compose.yml -p hitl-xgnn-fraud-detection logs -f

# 4. Stop containers
docker compose -f docker/docker-compose.yml -p hitl-xgnn-fraud-detection down
```

### Deployed Services
- **Interactive React 18 Dashboard**: `http://localhost:3000`
- **FastAPI Async Backend**: `http://localhost:8000/api/v1`
- **Interactive OpenAPI Docs**: `http://localhost:8000/api/v1/docs`
- **Backend Health Probe**: `http://localhost:8000/api/v1/health`
- **PostgreSQL Database**: `localhost:5432` (DB: `hitl_fraud_detection`)

---

## 12. Automated Testing Suite

The project includes complete unit and component test suites covering all backend modules, APIs, database operations, and frontend React views:

### Backend Test Suite (**76 tests across 12 files**)
```bash
.venv\Scripts\pytest tests/unit -v
```

Test coverage:
- `test_canonical_tx_ids.py`: Canonical ID resolution and 404 validation
- `test_docker_config.py`: Dockerfile and Nginx configuration validation
- `test_gnn_models.py`: GCN, GraphSAGE, GAT forward passes and loss backpropagation
- `test_module1_dataset.py`: Dataset loader and validation rules
- `test_module2_preprocessing.py`: Feature engineering, train-only scaling, temporal splits
- `test_module3_graph_builder.py`: PyG graph construction and 2-hop extraction
- `test_module4_baselines.py`: Tabular baseline training, Focal loss, threshold sweep
- `test_module5_explainability.py`: GNNExplainer attribution and fidelity calculations
- `test_module5_temporal_drift.py`: Period classification and drift aggregation
- `test_module6_api.py`: FastAPI routes, schemas, pagination, and OpenAPI contracts
- `test_module7_hitl.py`: Triage scoring, review protocols, active learning queries
- `test_module8_database.py`: SQLAlchemy models, migrations, CRUD, and seeding

### Frontend Test Suite (**9 tests across 6 suites**)
```bash
cd src/module6_api/frontend
npm run test
```

---

## 13. Research Limitations & Future Directions

### Limitations
1. **Elliptic Anonymization**: Elliptic transaction features are pre-extracted and PCA-transformed without raw Bitcoin addresses or wallet entity identities.
2. **Coarse Binned Timesteps**: Discrete 2-week timestep bins obscure sub-second transaction sequences.
3. **Simulated Reviewer Oracle**: Active learning experiments utilized simulated ground-truth annotations rather than real-world compliance analyst reviews.

### Future Work
1. **Dynamic / Continuous-Time Temporal GNNs**: Evaluating TGN, DyRep, or EvolveGCN on continuous timestamped transaction streams.
2. **Online Streaming Active Learning**: Implementing online streaming Bayesian updates for continuous feedback ingestion.
3. **Human Factor & Trust Studies**: Conducting user studies measuring analyst triage speed and decision accuracy with interactive GNNExplainer visualizations.

---

## 14. Phase 12 Completion Status

- **Phase 12.1 Audit**: Complete (Read-only verification of all 7 empirical experiments, 22 checkpoints, and 49 PyG graphs).
- **Phase 12.2 Documentation Freeze**: Complete (Canonical metrics frozen, `RESEARCH.md` paper compendium created, `REPRODUCIBILITY.md` runbook created, `README.md` updated, Docker compose cleaned).

