import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getHealth,
  getTransactions,
  getTransactionSubgraph,
  getTransactionExplanation,
  submitAnalystFeedback,
  ApiError,
} from '../api/client';

describe('API Client Layer Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('successfully fetches health probe', async () => {
    const mockHealth = {
      status: 'healthy',
      database_connected: true,
      model_available: true,
      active_model_version: 'graphsage_hitl_uncertainty_20',
      timestamp: new Date().toISOString(),
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    });

    const result = await getHealth();
    expect(result.status).toBe('healthy');
    expect(result.database_connected).toBe(true);
    expect(result.active_model_version).toBe('graphsage_hitl_uncertainty_20');
  });

  it('fetches transactions with filter parameters', async () => {
    const mockTxResponse = {
      items: [
        {
          tx_id: 'tx_101',
          timestep: 40,
          ground_truth_label: 1,
          predicted_prob: 0.89,
          risk_level: 'CRITICAL',
          is_triaged: false,
          triage_status: 'QUEUED',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ],
      total: 1,
      page: 1,
      page_size: 25,
      total_pages: 1,
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockTxResponse,
    });

    const result = await getTransactions({ timestep: 40, risk_level: 'CRITICAL', search: '10487903' });
    expect(result.total).toBe(1);
    expect(result.items[0].tx_id).toBe('tx_101');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/transactions?timestep=40&risk_level=CRITICAL&search=10487903'),
      expect.anything()
    );
  });

  it('handles API error responses correctly by throwing ApiError', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({
        error_code: 'HTTP_404',
        message: 'Transaction not found',
      }),
    });

    await expect(getTransactionSubgraph('invalid_tx')).rejects.toThrow(ApiError);
  });

  it('submits analyst feedback payload via POST', async () => {
    const mockFeedbackResponse = {
      id: 42,
      tx_id: 'tx_202',
      analyst_id: 'analyst_01',
      verdict: 'ILLICIT',
      confidence: 5,
      rationale: 'Confirmed mixing patterns',
      explanation_viewed: true,
      feedback_source: 'HUMAN_ANALYST',
      reviewed_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockFeedbackResponse,
    });

    const result = await submitAnalystFeedback({
      tx_id: 'tx_202',
      analyst_id: 'analyst_01',
      verdict: 'ILLICIT',
      confidence: 5,
      rationale: 'Confirmed mixing patterns',
      explanation_viewed: true,
    });

    expect(result.id).toBe(42);
    expect(result.verdict).toBe('ILLICIT');
  });
});
