/**
 * TypeScript definitions mapping to the FastAPI Pydantic schemas.
 */

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  database_connected: boolean;
  model_available: boolean;
  active_model_version?: string;
  architecture?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface TransactionResponse {
  tx_id: string;
  timestep: number;
  ground_truth_label: number; // -1 = unknown, 0 = licit, 1 = illicit
  predicted_prob?: number;
  predicted_class?: number;
  risk_level?: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  uncertainty_score?: number;
  entropy?: number;
  priority_score?: number;
  is_triaged: boolean;
  triage_status: 'PENDING' | 'QUEUED' | 'REVIEWED' | 'ESCALATED';
  created_at: string;
  updated_at: string;
}

export interface TransactionDetailResponse extends TransactionResponse {
  explanation_count: number;
  feedback_count: number;
  outgoing_edge_count: number;
  incoming_edge_count: number;
  model_version?: string;
  architecture?: string;
  decision_threshold?: number;
  decision_rule?: string;
  ground_truth_status?: 'known' | 'unavailable';
  outcome_category?: 'TRUE_POSITIVE' | 'FALSE_POSITIVE' | 'TRUE_NEGATIVE' | 'FALSE_NEGATIVE' | 'NOT_EVALUABLE';
  explanation_status?: 'AVAILABLE' | 'NOT_GENERATED';
  explanation_method?: string;
}

export interface GroupEvaluationResponse {
  scope: string;
  timestep?: number;
  total_transactions: number;
  labeled_count: number;
  unlabeled_count: number;
  illicit_count: number;
  licit_count: number;
  decision_threshold: number;
  f1_score?: number;
  pr_auc?: number;
  precision?: number;
  recall?: number;
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  status: string;
  notes: string;
}

export interface ThresholdPolicyResponse {
  active_model: string;
  architecture: string;
  calibrated_threshold: number;
  calibration_method: string;
  calibration_split: string;
  grid_specification: {
    min_threshold: number;
    max_threshold: number;
    num_points: number;
    step_size?: number;
  };
  validation_metrics: {
    f1_illicit: number;
    precision: number;
    recall: number;
  };
  test_metrics: {
    f1_illicit: number;
    pr_auc: number;
    precision: number;
    recall: number;
    roc_auc?: number;
  };
  comparison_thresholds: Record<string, number>;
  decision_rules: Record<string, string>;
}

export interface ModelBenchmarkItem {
  name: string;
  type: string;
  is_graph: boolean;
  features: string;
  calibrated_threshold: number;
  f1_illicit: number;
  pr_auc: number;
  precision: number;
  recall: number;
  roc_auc: number;
  runtime_seconds: number;
  parameters?: number;
}

export interface ModelValidationResponse {
  timestamp_utc: string;
  train_timesteps: number[];
  validation_timesteps: number[];
  test_timesteps: number[];
  zero_leakage_policy: string;
  models: ModelBenchmarkItem[];
  verification_evidence: {
    verification_status: string;
    unit_test_suite: string;
    checkpoints: Record<string, { path: string; exists: boolean; size_bytes: number }>;
    dataset_invariants: {
      zero_orphan_nodes: boolean;
      zero_self_loops: boolean;
      intra_timestep_edges_pct: number;
      cross_timestep_edges: number;
    };
  };
}

export interface GraphNode {
  id: string;
  label: string;
  timestep: number;
  predicted_prob?: number;
  predicted_class?: number;
  risk_level?: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  ground_truth: number;
  is_target: boolean;
  in_degree: number;
  out_degree: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  timestep: number;
  importance_weight?: number;
  rank?: number;
}

export interface SubgraphResponse {
  target_tx_id: string;
  timestep: number;
  k_hops: number;
  num_nodes: number;
  num_edges: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  explanation_available: boolean;
}

export interface FeatureAttributionItem {
  feature_index: number;
  feature_name: string;
  feature_type: 'local' | 'aggregated';
  raw_importance: number;
  normalized_importance: number;
  rank: number;
}

export interface EdgeAttributionItem {
  source_tx_id: string;
  target_tx_id: string;
  source_global_idx: number;
  target_global_idx: number;
  edge_index_in_subgraph: number;
  raw_importance: number;
  normalized_importance: number;
  rank: number;
}

export interface FidelityBenchmarkMethod {
  mean_fidelity_plus: number;
  mean_fidelity_minus: number;
  median_fidelity_plus?: number;
  median_fidelity_minus?: number;
  std_fidelity_plus?: number;
  std_fidelity_minus?: number;
}

export interface FidelityBenchmarkSummaryResponse {
  overall_fidelity: Record<string, FidelityBenchmarkMethod>;
  category_fidelity?: Record<string, { mean_fid_plus: number; mean_fid_minus: number; count: number }>;
  regime_fidelity?: Record<string, { mean_fid_plus: number; mean_fid_minus: number }>;
  latency_profile?: {
    num_samples: number;
    mean_latency_ms: number;
    median_latency_ms: number;
    p95_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
    total_duration_seconds: number;
  };
}

export interface ExplanationResponse {
  tx_id: string;
  timestep: number;
  model_version: string;
  prediction_probability: number;
  predicted_class: number;
  risk_level: string;
  category: string;
  period: string;
  subgraph_num_nodes: number;
  subgraph_num_edges: number;
  fidelity_plus?: number;
  fidelity_minus?: number;
  edge_sparsity?: number;
  feature_sparsity?: number;
  generation_latency_ms: number;
  top_features: FeatureAttributionItem[];
  top_edges: EdgeAttributionItem[];
  feature_summary: Record<string, any>;
  benchmark_summary?: FidelityBenchmarkSummaryResponse;
}

export interface FeedbackSubmissionRequest {
  tx_id: string;
  analyst_id: string;
  verdict: 'ILLICIT' | 'LICIT' | 'ESCALATED' | 'INCONCLUSIVE';
  confidence: number; // 1 to 5
  rationale: string;
  explanation_viewed: boolean;
  feedback_source?: 'HUMAN_ANALYST' | 'SIMULATED_ORACLE';
}

export interface FeedbackResponse {
  id: number;
  tx_id: string;
  analyst_id: string;
  verdict: 'ILLICIT' | 'LICIT' | 'ESCALATED' | 'INCONCLUSIVE';
  confidence: number;
  rationale: string;
  model_predicted_score?: number;
  explanation_viewed: boolean;
  feedback_source: string;
  reviewed_at: string;
  created_at: string;
}

export interface TriageQueueItem {
  tx_id: string;
  timestep: number;
  predicted_prob: number;
  predicted_class: number;
  risk_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  uncertainty_score: number;
  entropy: number;
  priority_score: number;
  reason: string;
  triage_status: string;
}

export interface TriageQueueResponse {
  items: TriageQueueItem[];
  total: number;
  min_priority: number;
  risk_distribution: Record<string, number>;
}

export interface TemporalMetricItem {
  timestep: number;
  period: string;
  num_labeled: number;
  num_illicit: number;
  illicit_prevalence_pct: number;
  f1_score: number;
  pr_auc: number;
  precision: number;
  recall: number;
}

export interface ModelPerformanceSummary {
  model_version: string;
  architecture: string;
  decision_threshold: number;
  test_f1: number;
  test_pr_auc: number;
  test_precision: number;
  test_recall: number;
  test_roc_auc: number;
}

export interface AnalyticsMetricsResponse {
  dataset_summary: {
    total_transactions: number;
    total_edges: number;
    total_timesteps: number;
    total_licit: number;
    total_illicit: number;
    total_unlabeled: number;
    imbalance_ratio: number;
  };
  active_model: ModelPerformanceSummary;
  temporal_drift: TemporalMetricItem[];
  active_learning_benchmarks: Record<string, any>;
  feedback_stats: Record<string, number>;
}
