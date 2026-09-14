import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ExplainabilityPanel } from '../components/explainability/ExplainabilityPanel';

describe('Explainability Panel Component', () => {
  it('renders feature attributions and fidelity metrics', async () => {
    const mockExplanation = {
      tx_id: 'tx_expl_target',
      timestep: 40,
      model_version: 'graphsage_hitl_uncertainty_20',
      prediction_probability: 0.915,
      predicted_class: 1,
      risk_level: 'CRITICAL',
      category: 'TP',
      period: 'pre_shock',
      subgraph_num_nodes: 5,
      subgraph_num_edges: 4,
      fidelity_plus: 0.2845,
      fidelity_minus: 0.012,
      edge_sparsity: 0.82,
      feature_sparsity: 0.9,
      generation_latency_ms: 245.8,
      top_features: [
        {
          feature_index: 12,
          feature_name: 'feat_12_out_trans_vol_std',
          feature_type: 'local',
          raw_importance: 0.85,
          normalized_importance: 0.15,
          rank: 1,
        },
      ],
      top_edges: [
        {
          source_tx_id: 'tx_expl_target',
          target_tx_id: 'tx_partner_99',
          source_global_idx: 0,
          target_global_idx: 1,
          edge_index_in_subgraph: 0,
          raw_importance: 0.92,
          normalized_importance: 0.35,
          rank: 1,
        },
      ],
      feature_summary: {},
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockExplanation,
    });

    render(
      <ExplainabilityPanel
        txId="tx_expl_target"
        onNavigateTab={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('tx_expl_target')).toBeInTheDocument();
      expect(screen.getByText('+0.2845')).toBeInTheDocument(); // Fidelity+
      expect(screen.getByText('feat_12_out_trans_vol_std')).toBeInTheDocument();
      expect(screen.getByText('245.8 ms')).toBeInTheDocument();
    });
  });
});
