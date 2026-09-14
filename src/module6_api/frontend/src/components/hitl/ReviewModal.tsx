import React, { useState } from 'react';
import { X, ShieldAlert, CheckCircle, AlertTriangle, HelpCircle, Send } from 'lucide-react';
import { submitAnalystFeedback } from '../../api/client';
import { FeedbackResponse } from '../../types/api';
import { Badge } from '../common/Badge';

interface ReviewModalProps {
  txId: string;
  timestep?: number;
  predictedProb?: number;
  riskLevel?: string;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (feedback: FeedbackResponse) => void;
}

export const ReviewModal: React.FC<ReviewModalProps> = ({
  txId,
  timestep = 40,
  predictedProb = 0.85,
  riskLevel = 'CRITICAL',
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [analystId, setAnalystId] = useState('analyst_01');
  const [verdict, setVerdict] = useState<'ILLICIT' | 'LICIT' | 'ESCALATED' | 'INCONCLUSIVE'>('ILLICIT');
  const [confidence, setConfidence] = useState<number>(5);
  const [rationale, setRationale] = useState('');
  const [explanationViewed, setExplanationViewed] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rationale.trim()) {
      setError('Please provide an investigation rationale / case justification.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const res = await submitAnalystFeedback({
        tx_id: txId,
        analyst_id: analystId,
        verdict: verdict,
        confidence: confidence,
        rationale: rationale,
        explanation_viewed: explanationViewed,
        feedback_source: 'HUMAN_ANALYST',
      });
      onSuccess(res);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to submit analyst feedback.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <ShieldAlert size={20} style={{ color: '#06b6d4' }} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc' }}>
              Submit Analyst Investigation Verdict
            </h3>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {error && (
            <div
              style={{
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid #ef4444',
                borderRadius: '6px',
                padding: '10px 14px',
                fontSize: '0.8125rem',
                color: '#f87171',
              }}
            >
              {error}
            </div>
          )}

          {/* Transaction Metadata Preview */}
          <div
            style={{
              backgroundColor: '#162032',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '12px 14px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '0.8125rem',
            }}
          >
            <div>
              <div style={{ color: '#64748b', fontSize: '0.7rem' }}>TRANSACTION ID</div>
              <div className="font-mono" style={{ color: '#38bdf8', fontWeight: 600 }}>{txId}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ color: '#64748b', fontSize: '0.7rem' }}>MODEL RISK / PROB</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Badge type="risk" value={riskLevel} />
                <span style={{ fontWeight: 600 }}>{(predictedProb * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Analyst ID */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Analyst Identifier
            </label>
            <input
              type="text"
              value={analystId}
              onChange={(e) => setAnalystId(e.target.value)}
              required
              style={{
                width: '100%',
                backgroundColor: '#1e293b',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '8px 12px',
                color: '#f8fafc',
                fontSize: '0.875rem',
              }}
            />
          </div>

          {/* Verdict Selection */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Final Case Verdict
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
              {(['ILLICIT', 'LICIT', 'ESCALATED', 'INCONCLUSIVE'] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setVerdict(v)}
                  style={{
                    padding: '8px',
                    borderRadius: '6px',
                    border: '1px solid',
                    borderColor: verdict === v ? '#06b6d4' : 'var(--border-color)',
                    backgroundColor: verdict === v ? 'rgba(6, 182, 212, 0.2)' : '#1e293b',
                    color: verdict === v ? '#38bdf8' : '#cbd5e1',
                    fontSize: '0.8125rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                  }}
                >
                  {v === 'ILLICIT' && <ShieldAlert size={14} style={{ color: '#ef4444' }} />}
                  {v === 'LICIT' && <CheckCircle size={14} style={{ color: '#10b981' }} />}
                  {v === 'ESCALATED' && <AlertTriangle size={14} style={{ color: '#f97316' }} />}
                  {v === 'INCONCLUSIVE' && <HelpCircle size={14} style={{ color: '#fbbf24' }} />}
                  {v}
                </button>
              ))}
            </div>
          </div>

          {/* Confidence Score (1–5) */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.8125rem' }}>
              <label style={{ fontWeight: 600, color: '#cbd5e1' }}>Confidence Level</label>
              <span style={{ color: '#38bdf8', fontWeight: 600 }}>{confidence} / 5</span>
            </div>
            <input
              type="range"
              min={1}
              max={5}
              value={confidence}
              onChange={(e) => setConfidence(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: '#06b6d4' }}
            />
          </div>

          {/* Rationale Textarea */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Investigation Rationale & Notes
            </label>
            <textarea
              rows={3}
              placeholder="Detail the topological mixing pattern or verified licit exchange proof..."
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              required
              style={{
                width: '100%',
                backgroundColor: '#1e293b',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '8px 12px',
                color: '#f8fafc',
                fontSize: '0.875rem',
                outline: 'none',
                resize: 'vertical',
              }}
            />
          </div>

          {/* Explanation Viewed Checkbox */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="checkbox"
              id="explCheck"
              checked={explanationViewed}
              onChange={(e) => setExplanationViewed(e.target.checked)}
              style={{ accentColor: '#06b6d4' }}
            />
            <label htmlFor="explCheck" style={{ fontSize: '0.8125rem', color: '#94a3b8', cursor: 'pointer' }}>
              Corroborated with GNNExplainer 2-hop attribution subgraph
            </label>
          </div>

          {/* Modal Footer */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: '10px',
              paddingTop: '12px',
              borderTop: '1px solid var(--border-color)',
            }}
          >
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="btn btn-primary"
            >
              <Send size={14} /> {submitting ? 'Recording...' : 'Submit Verdict'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
