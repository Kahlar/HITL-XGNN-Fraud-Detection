import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { OverviewDashboard } from '../components/overview/OverviewDashboard';

describe('Overview Dashboard Component', () => {
  it('renders overview metrics from API response', async () => {
    const mockMetrics = {
      dataset_summary: {
        total_transactions: 203769,
        total_edges: 234355,
        total_timesteps: 49,
        total_licit: 42019,
        total_illicit: 4545,
        total_unlabeled: 157205,
        imbalance_ratio: 9.25,
      },
      active_model: {
        model_version: 'graphsage_hitl_uncertainty_20',
        architecture: 'GraphSAGE',
        decision_threshold: 0.5517,
        test_f1: 0.503,
        test_pr_auc: 0.4358,
        test_precision: 0.7086,
        test_recall: 0.3899,
        test_roc_auc: 0.825,
      },
      temporal_drift: [
        {
          timestep: 40,
          period: 'pre_shock',
          num_labeled: 1000,
          num_illicit: 80,
          illicit_prevalence_pct: 8.0,
          f1_score: 0.65,
          pr_auc: 0.55,
          precision: 0.75,
          recall: 0.58,
        },
      ],
      active_learning_benchmarks: {},
      feedback_stats: {},
    };

    const mockQueue = {
      items: [],
      total: 12,
      min_priority: 0.4,
      risk_distribution: { CRITICAL: 4, HIGH: 8 },
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/analytics/metrics')) {
        return Promise.resolve({ ok: true, json: async () => mockMetrics });
      }
      if (url.includes('/hitl/queue')) {
        return Promise.resolve({ ok: true, json: async () => mockQueue });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<OverviewDashboard onNavigate={() => {}} />);

    // Check loading text first
    expect(screen.getByText(/Loading fraud intelligence metrics/i)).toBeInTheDocument();

    // Wait for metrics to load
    await waitFor(() => {
      expect(screen.getByText('203,769')).toBeInTheDocument();
      expect(screen.getByText('4,545')).toBeInTheDocument();
      expect(screen.getByText('0.5030')).toBeInTheDocument();
      expect(screen.getByText('12')).toBeInTheDocument();
    });
  });
});
