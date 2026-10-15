import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PredictionDetailPanel } from '../components/explorer/PredictionDetailPanel';
import { AnalyticsView } from '../components/analytics/AnalyticsView';

describe('Prediction Details and Model Validation UI Tests', () => {
  it('renders PredictionDetailPanel with telemetry, threshold rule, and group metrics', async () => {
    const mockTx = {
      tx_id: '10487903',
      timestep: 40,
      ground_truth_label: -1,
      predicted_prob: 0.8841,
      predicted_class: 1,
      risk_level: 'CRITICAL' as const,
      uncertainty_score: 0.2318,
      entropy: 0.5132,
      priority_score: 0.85,
      is_triaged: false,
      triage_status: 'QUEUED' as const,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    const mockDetail = {
      ...mockTx,
      explanation_count: 1,
      feedback_count: 0,
      outgoing_edge_count: 0,
      incoming_edge_count: 0,
      model_version: 'graphsage_hitl_uncertainty_20',
      architecture: 'GraphSAGE',
      decision_threshold: 0.5517,
      decision_rule: 'p >= 0.5517 -> Illicit (1); otherwise Licit (0)',
      ground_truth_status: 'unavailable' as const,
      outcome_category: 'NOT_EVALUABLE' as const,
      explanation_status: 'AVAILABLE' as const,
      explanation_method: 'GNNExplainer',
    };

    const mockGroupEval = {
      scope: 'timestep',
      timestep: 40,
      total_transactions: 2000,
      labeled_count: 500,
      unlabeled_count: 1500,
      illicit_count: 80,
      licit_count: 420,
      decision_threshold: 0.5517,
      f1_score: 0.5175,
      pr_auc: 0.4499,
      precision: 0.7456,
      recall: 0.3962,
      true_positives: 32,
      false_positives: 11,
      true_negatives: 409,
      false_negatives: 48,
      status: 'VALID',
      notes: 'Evaluated strictly over 500 labeled nodes.',
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/evaluation-metrics')) {
        return Promise.resolve({ ok: true, json: async () => mockGroupEval });
      }
      return Promise.resolve({ ok: true, json: async () => mockDetail });
    });

    render(
      <PredictionDetailPanel
        tx={mockTx}
        onClose={() => {}}
        onNavigateTab={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/10487903/)).toBeInTheDocument();
      expect(screen.getByText(/0\.8841/)).toBeInTheDocument();
      expect(screen.getByText('Illicit (1)')).toBeInTheDocument();
      expect(screen.getByText(/Unavailable/)).toBeInTheDocument();
      expect(screen.getByText(/Not evaluable/)).toBeInTheDocument();
      expect(screen.getByText(/0\.2318/)).toBeInTheDocument();
      expect(screen.getByText(/0\.5175/)).toBeInTheDocument();
      expect(screen.getByText(/0\.4499/)).toBeInTheDocument();
    });
  });

  it('renders AnalyticsView with Chronological Split Protocol and 7-model comparison table', async () => {
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
        test_f1: 0.5175,
        test_pr_auc: 0.4499,
        test_precision: 0.7456,
        test_recall: 0.3962,
        test_roc_auc: 0.8289,
      },
      temporal_drift: [],
      active_learning_benchmarks: {},
      feedback_stats: {},
    };

    const mockVal = {
      timestamp_utc: '2026-08-31T19:22:59Z',
      train_timesteps: [1, 34],
      validation_timesteps: [35, 39],
      test_timesteps: [40, 49],
      zero_leakage_policy: 'Strict chronological partitioning: zero leakage.',
      models: [
        {
          name: 'Xgboost',
          type: 'Tabular Baseline',
          is_graph: false,
          features: 'original_all (165)',
          calibrated_threshold: 0.9014,
          f1_illicit: 0.7223,
          pr_auc: 0.6712,
          precision: 0.9608,
          recall: 0.5786,
          roc_auc: 0.8828,
          runtime_seconds: 3.3,
        },
        {
          name: 'GRAPHSAGE',
          type: 'Graph Neural Network',
          is_graph: true,
          features: 'original_all (165)',
          calibrated_threshold: 0.5517,
          f1_illicit: 0.5175,
          pr_auc: 0.4499,
          precision: 0.7456,
          recall: 0.3962,
          roc_auc: 0.8289,
          runtime_seconds: 42.83,
        },
      ],
      verification_evidence: {
        verification_status: 'VERIFIED',
        unit_test_suite: 'PASSING',
        checkpoints: {},
        dataset_invariants: {
          zero_orphan_nodes: true,
          zero_self_loops: true,
          intra_timestep_edges_pct: 100.0,
          cross_timestep_edges: 0,
        },
      },
    };

    const mockThresh = {
      active_model: 'graphsage_hitl_uncertainty_20',
      architecture: 'GraphSAGE',
      calibrated_threshold: 0.5517,
      calibration_method: 'Minority F1 Grid Search',
      calibration_split: 'Val t in [35, 39]',
      grid_specification: { min_threshold: 0.01, max_threshold: 0.99, num_points: 200 },
      validation_metrics: { f1_illicit: 0.7299, precision: 0.7712, recall: 0.6928 },
      test_metrics: { f1_illicit: 0.5175, pr_auc: 0.4499, precision: 0.7456, recall: 0.3962 },
      comparison_thresholds: { xgboost: 0.9014, graphsage: 0.5517 },
      decision_rules: {},
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/models/validation')) {
        return Promise.resolve({ ok: true, json: async () => mockVal });
      }
      if (url.includes('/models/threshold')) {
        return Promise.resolve({ ok: true, json: async () => mockThresh });
      }
      return Promise.resolve({ ok: true, json: async () => mockMetrics });
    });

    render(<AnalyticsView />);

    await waitFor(() => {
      expect(screen.getByText('Model Validation & Verification Methodology')).toBeInTheDocument();
      expect(screen.getByText(/Chronological Splitting Protocol/i)).toBeInTheDocument();
      expect(screen.getByText('TRAINING PARTITION (t = 1 .. 34)')).toBeInTheDocument();
      expect(screen.getByText('VALIDATION PARTITION (t = 35 .. 39)')).toBeInTheDocument();
      expect(screen.getByText('TEST PARTITION (t = 40 .. 49)')).toBeInTheDocument();
      expect(screen.getByText('Xgboost')).toBeInTheDocument();
    });
  });
});
