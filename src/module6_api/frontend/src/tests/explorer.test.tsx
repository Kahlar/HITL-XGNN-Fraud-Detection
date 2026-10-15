import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TransactionExplorer } from '../components/explorer/TransactionExplorer';

describe('Transaction Explorer Component', () => {
  it('renders transactions and filters', async () => {
    const mockTxList = {
      items: [
        {
          tx_id: 'tx_alpha_999',
          timestep: 42,
          ground_truth_label: 1,
          predicted_prob: 0.94,
          risk_level: 'CRITICAL',
          uncertainty_score: 0.38,
          priority_score: 0.88,
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
      json: async () => mockTxList,
    });

    render(
      <TransactionExplorer
        onSelectTx={() => {}}
        onNavigateTab={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('tx_alpha_999')).toBeInTheDocument();
      expect(screen.getByText('CRITICAL')).toBeInTheDocument();
      expect(screen.getByText('TS #42')).toBeInTheDocument();
    });
  });

  it('triggers search query with debouncing when typing in search bar', async () => {
    const mockTxList = {
      items: [
        {
          tx_id: '10487903',
          timestep: 40,
          ground_truth_label: 0,
          predicted_prob: 0.12,
          risk_level: 'LOW',
          uncertainty_score: 0.15,
          priority_score: 0.10,
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

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockTxList,
    });
    globalThis.fetch = fetchMock;

    const { fireEvent } = await import('@testing-library/react');
    render(
      <TransactionExplorer
        onSelectTx={() => {}}
        onNavigateTab={() => {}}
      />
    );

    const input = screen.getByPlaceholderText('Search Transaction ID...');
    fireEvent.change(input, { target: { value: '10487903' } });

    await waitFor(
      () => {
        expect(fetchMock).toHaveBeenCalledWith(
          expect.stringContaining('search=10487903'),
          expect.anything()
        );
      },
      { timeout: 1500 }
    );
  });

  it('handles partial search and empty state when no transactions match', async () => {
    const emptyResponse = {
      items: [],
      total: 0,
      page: 1,
      page_size: 25,
      total_pages: 0,
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => emptyResponse,
    });
    globalThis.fetch = fetchMock;

    const { fireEvent } = await import('@testing-library/react');
    render(
      <TransactionExplorer
        onSelectTx={() => {}}
        onNavigateTab={() => {}}
      />
    );

    const input = screen.getByPlaceholderText('Search Transaction ID...');
    fireEvent.change(input, { target: { value: 'nonexistent_99999' } });

    await waitFor(() => {
      expect(screen.getByText('No transactions match the selected filters')).toBeInTheDocument();
    });
  });

  it('renders transactions with null or missing fields without throwing errors', async () => {
    const mockTxListWithNulls = {
      items: [
        {
          tx_id: '10487903',
          timestep: 40,
          ground_truth_label: 0,
          predicted_prob: null,
          risk_level: null,
          uncertainty_score: null,
          priority_score: null,
          is_triaged: true,
          triage_status: 'REVIEWED',
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
      json: async () => mockTxListWithNulls,
    });

    render(
      <TransactionExplorer
        onSelectTx={() => {}}
        onNavigateTab={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('10487903')).toBeInTheDocument();
      expect(screen.getByText('TS #40')).toBeInTheDocument();
      expect(screen.getByText('REVIEWED')).toBeInTheDocument();
    });
  });
});
