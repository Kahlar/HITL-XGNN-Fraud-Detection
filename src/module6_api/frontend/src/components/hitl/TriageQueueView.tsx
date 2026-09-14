import React, { useEffect, useState } from 'react';
import { getTriageQueue, TriageQueueParams } from '../../api/client';
import { FeedbackResponse, TriageQueueItem, TriageQueueResponse } from '../../types/api';
import { TriageQueueTable } from './TriageQueueTable';
import { ReviewModal } from './ReviewModal';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorAlert } from '../common/ErrorAlert';
import { EmptyState } from '../common/EmptyState';
import { ShieldCheck, RefreshCw, Layers } from 'lucide-react';

interface TriageQueueViewProps {
  onNavigateTab: (tab: any, txId: string) => void;
}

export const TriageQueueView: React.FC<TriageQueueViewProps> = ({ onNavigateTab }) => {
  const [params, setParams] = useState<TriageQueueParams>({
    min_priority: 0.40,
    limit: 50,
  });
  const [queueData, setQueueData] = useState<TriageQueueResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedReviewTx, setSelectedReviewTx] = useState<TriageQueueItem | null>(null);
  const [successToast, setSuccessToast] = useState<string | null>(null);

  const fetchQueue = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getTriageQueue(params);
      setQueueData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load triage queue');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, [params]);

  const handleReviewSuccess = (feedback: FeedbackResponse) => {
    setSuccessToast(
      `Successfully recorded verdict '${feedback.verdict}' by ${feedback.analyst_id} for transaction ${feedback.tx_id}.`
    );
    fetchQueue();
    setTimeout(() => setSuccessToast(null), 6000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Toast Notification */}
      {successToast && (
        <div
          style={{
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            border: '1px solid #10b981',
            borderRadius: '6px',
            padding: '12px 16px',
            color: '#34d399',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.875rem',
            fontWeight: 500,
          }}
        >
          <ShieldCheck size={18} />
          <span>{successToast}</span>
        </div>
      )}

      {/* Header & Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
            Human-in-the-Loop Triage Review Queue
          </h2>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            Prioritized cases ranked by composite uncertainty margin, high fraud risk, and graph topological diversity.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Priority Slider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem' }}>
            <span style={{ color: '#94a3b8' }}>Min Priority:</span>
            <span style={{ color: '#fbbf24', fontWeight: 700 }}>
              {((params.min_priority ?? 0.40) * 100).toFixed(0)}%
            </span>
            <input
              type="range"
              min={0}
              max={100}
              value={(params.min_priority ?? 0.40) * 100}
              onChange={(e) =>
                setParams((prev) => ({ ...prev, min_priority: parseInt(e.target.value, 10) / 100.0 }))
              }
              style={{ accentColor: '#f59e0b', width: '100px' }}
            />
          </div>

          <button onClick={fetchQueue} className="btn btn-secondary" style={{ padding: '8px 12px' }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Risk Distribution Summary Bar */}
      {queueData && (
        <div
          style={{
            display: 'flex',
            gap: '16px',
            backgroundColor: '#0f172a',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '12px 16px',
            fontSize: '0.8125rem',
          }}
        >
          <div style={{ color: '#94a3b8' }}>Queue Composition:</div>
          <div>
            Critical: <strong style={{ color: '#f87171' }}>{queueData.risk_distribution?.CRITICAL || 0}</strong>
          </div>
          <div>
            High: <strong style={{ color: '#fb923c' }}>{queueData.risk_distribution?.HIGH || 0}</strong>
          </div>
          <div>
            Medium: <strong style={{ color: '#fbbf24' }}>{queueData.risk_distribution?.MEDIUM || 0}</strong>
          </div>
          <div>
            Low: <strong style={{ color: '#34d399' }}>{queueData.risk_distribution?.LOW || 0}</strong>
          </div>
        </div>
      )}

      {/* Queue Content */}
      {loading ? (
        <LoadingSpinner message="Calculating active triage priority rankings..." />
      ) : error ? (
        <ErrorAlert message={error} onRetry={fetchQueue} />
      ) : !queueData || queueData.items.length === 0 ? (
        <EmptyState title="No transactions meet the priority threshold" />
      ) : (
        <TriageQueueTable
          items={queueData.items}
          onOpenReview={(item) => setSelectedReviewTx(item)}
          onNavigateTab={onNavigateTab}
        />
      )}

      {/* Review Modal */}
      {selectedReviewTx && (
        <ReviewModal
          txId={selectedReviewTx.tx_id}
          timestep={selectedReviewTx.timestep}
          predictedProb={selectedReviewTx.predicted_prob}
          riskLevel={selectedReviewTx.risk_level}
          isOpen={true}
          onClose={() => setSelectedReviewTx(null)}
          onSuccess={handleReviewSuccess}
        />
      )}
    </div>
  );
};
