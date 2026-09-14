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
