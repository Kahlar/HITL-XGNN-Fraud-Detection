/**
 * Centralized API client for communicating with the FastAPI backend.
 */

import {
  AnalyticsMetricsResponse,
  ExplanationResponse,
  FeedbackResponse,
  FeedbackSubmissionRequest,
  HealthResponse,
  PaginatedResponse,
  SubgraphResponse,
  TransactionDetailResponse,
  TransactionResponse,
  TriageQueueResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export class ApiError extends Error {
  public statusCode: number;
  public errorCode: string;
  public details?: any;

  constructor(message: string, statusCode: number, errorCode: string = 'API_ERROR', details?: any) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.details = details;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...options.headers,
  };

  try {
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      let errorBody: any;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = { message: response.statusText };
      }
      throw new ApiError(
        errorBody.message || `Request failed with status ${response.status}`,
        response.status,
        errorBody.error_code || `HTTP_${response.status}`,
        errorBody.details
      );
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Network error: Could not reach the API service',
      0,
      'NETWORK_ERROR'
    );
  }
}

// 1. Health Probe
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

// 2. Transactions
export interface TransactionFilterParams {
  page?: number;
  page_size?: number;
  timestep?: number;
  risk_level?: string;
  triage_status?: string;
  min_priority?: number;
  search?: string;
}

export async function getTransactions(params: TransactionFilterParams = {}): Promise<PaginatedResponse<TransactionResponse>> {
  const query = new URLSearchParams();
  if (params.page) query.append('page', params.page.toString());
  if (params.page_size) query.append('page_size', params.page_size.toString());
  if (params.timestep) query.append('timestep', params.timestep.toString());
  if (params.risk_level) query.append('risk_level', params.risk_level);
  if (params.triage_status) query.append('triage_status', params.triage_status);
  if (params.min_priority !== undefined) query.append('min_priority', params.min_priority.toString());
  if (params.search) query.append('search', params.search.trim());

  const queryString = query.toString() ? `?${query.toString()}` : '';
  return request<PaginatedResponse<TransactionResponse>>(`/transactions${queryString}`);
}

export async function getTransactionDetail(txId: string): Promise<TransactionDetailResponse> {
  return request<TransactionDetailResponse>(`/transactions/${encodeURIComponent(txId)}`);
}

// 3. Subgraph Visualizer
export async function getTransactionSubgraph(txId: string, hops: number = 2, timestep?: number): Promise<SubgraphResponse> {
  const query = new URLSearchParams({ hops: hops.toString() });
  if (timestep) query.append('timestep', timestep.toString());
  return request<SubgraphResponse>(`/graph/${encodeURIComponent(txId)}/subgraph?${query.toString()}`);
}

// 4. Explainability Engine
export async function getTransactionExplanation(txId: string, timestep?: number): Promise<ExplanationResponse> {
  const query = timestep ? `?timestep=${timestep}` : '';
  return request<ExplanationResponse>(`/explain/${encodeURIComponent(txId)}${query}`);
}

// 5. Human-in-the-Loop
export async function submitAnalystFeedback(payload: FeedbackSubmissionRequest): Promise<FeedbackResponse> {
  return request<FeedbackResponse>('/hitl/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export interface TriageQueueParams {
  timestep?: number;
  min_priority?: number;
  limit?: number;
  offset?: number;
}

export async function getTriageQueue(params: TriageQueueParams = {}): Promise<TriageQueueResponse> {
  const query = new URLSearchParams();
  if (params.timestep) query.append('timestep', params.timestep.toString());
  if (params.min_priority !== undefined) query.append('min_priority', params.min_priority.toString());
  if (params.limit) query.append('limit', params.limit.toString());
  if (params.offset) query.append('offset', params.offset.toString());

  const queryString = query.toString() ? `?${query.toString()}` : '';
  return request<TriageQueueResponse>(`/hitl/queue${queryString}`);
}

// 6. Analytics Telemetry
export async function getAnalyticsMetrics(): Promise<AnalyticsMetricsResponse> {
  return request<AnalyticsMetricsResponse>('/analytics/metrics');
}
