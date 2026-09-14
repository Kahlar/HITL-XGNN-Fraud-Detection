import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TriageQueueView } from '../components/hitl/TriageQueueView';

describe('HITL Triage Queue & Review Modal Component', () => {
  it('renders triage queue and opens review modal on action click', async () => {
    const mockQueue = {
      items: [
        {
          tx_id: 'tx_triage_001',
          timestep: 45,
          predicted_prob: 0.88,
          predicted_class: 1,
          risk_level: 'CRITICAL',
          uncertainty_score: 0.35,
          entropy: 0.52,
          priority_score: 0.89,
          reason: 'High risk uncertainty candidate',
          triage_status: 'QUEUED',
        },
      ],
      total: 1,
      min_priority: 0.4,
      risk_distribution: { CRITICAL: 1 },
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string, options?: any) => {
      if (options && options.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: 99,
            tx_id: 'tx_triage_001',
            analyst_id: 'analyst_01',
            verdict: 'ILLICIT',
            confidence: 5,
            rationale: 'Confirmed',
            explanation_viewed: true,
            feedback_source: 'HUMAN_ANALYST',
            reviewed_at: new Date().toISOString(),
            created_at: new Date().toISOString(),
          }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => mockQueue });
    });

    render(<TriageQueueView onNavigateTab={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText('tx_triage_001')).toBeInTheDocument();
      expect(screen.getByText('89.0%')).toBeInTheDocument(); // priority
    });

    // Click "Audit Case" button
    const auditBtn = screen.getByText(/Audit Case/i);
    fireEvent.click(auditBtn);

    // Verify modal is open
    expect(screen.getByText(/Submit Analyst Investigation Verdict/i)).toBeInTheDocument();
  });
});
