import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { InteractiveGraphView } from '../components/graph/InteractiveGraphView';

describe('Interactive Graph View Component', () => {
  it('renders graph canvas and nodes from API response', async () => {
    const mockSubgraph = {
      target_tx_id: 'tx_center_01',
      timestep: 40,
      k_hops: 2,
      num_nodes: 3,
      num_edges: 2,
      nodes: [
        {
          id: 'tx_center_01',
          label: 'tx_center...',
          timestep: 40,
          predicted_prob: 0.92,
          risk_level: 'CRITICAL',
          ground_truth: 1,
          is_target: true,
          in_degree: 1,
          out_degree: 1,
        },
        {
          id: 'tx_neighbor_02',
          label: 'tx_neighb...',
          timestep: 40,
          predicted_prob: 0.45,
          risk_level: 'MEDIUM',
          ground_truth: 0,
          is_target: false,
          in_degree: 1,
          out_degree: 0,
        },
      ],
      edges: [
        {
          id: 'tx_center_01->tx_neighbor_02',
          source: 'tx_center_01',
          target: 'tx_neighbor_02',
          timestep: 40,
          importance_weight: 0.85,
        },
      ],
      explanation_available: true,
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockSubgraph,
    });

    render(
      <InteractiveGraphView
        initialTxId="tx_center_01"
        onNavigateTab={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Interactive 2-Hop Computational Graph Canvas/i)).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // numNodes
      expect(screen.getByText('2')).toBeInTheDocument(); // numEdges
    });
  });
});
