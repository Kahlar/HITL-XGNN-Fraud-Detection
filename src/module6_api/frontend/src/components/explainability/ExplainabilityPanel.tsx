import React, { useEffect, useState } from 'react';
import { getTransactionExplanation } from '../../api/client';
import { ExplanationResponse } from '../../types/api';
import { FidelityCard } from './FidelityCard';
import { FeatureAttributionTable } from './FeatureAttributionTable';
import { EdgeAttributionTable } from './EdgeAttributionTable';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorAlert } from '../common/ErrorAlert';
import { Badge } from '../common/Badge';
import { HelpCircle, Network, ArrowLeft } from 'lucide-react';

interface ExplainabilityPanelProps {
  txId: string;
  onNavigateTab: (tab: any, txId: string) => void;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  txId,
  onNavigateTab,
}) => {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchExplanationData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getTransactionExplanation(txId);
      setExplanation(res);
    } catch (err: any) {
      setError(err.message || `Failed to explain transaction ${txId}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (txId) {
      fetchExplanationData();
    }
  }, [txId]);

  if (loading) return <LoadingSpinner message={`Running GNNExplainer attribution for ${txId}...`} size="lg" />;
  if (error) return <ErrorAlert message={error} onRetry={fetchExplanationData} />;
  if (!explanation) return <div style={{ color: '#94a3b8' }}>No explanation available for transaction {txId}.</div>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Info */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <button
              onClick={() => onNavigateTab('explorer', txId)}
              className="btn btn-secondary"
              style={{ padding: '4px 8px', fontSize: '0.75rem' }}
            >
              <ArrowLeft size={13} /> Explorer
            </button>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
              XAI Explanation Report
            </h2>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.875rem' }}>
            <span style={{ color: '#64748b' }}>Target Transaction:</span>
            <span className="font-mono" style={{ color: '#38bdf8', fontWeight: 600 }}>{explanation.tx_id}</span>
            <span style={{ color: '#64748b' }}>Timestep #{explanation.timestep}</span>
            <Badge type="risk" value={explanation.risk_level} />
          </div>
        </div>

        <button
          onClick={() => onNavigateTab('graph', txId)}
          className="btn btn-primary"
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8125rem' }}
        >
          <Network size={15} /> View in Subgraph
        </button>
      </div>

      {/* Model vs Baseline Banner */}
      <div
        style={{
          backgroundColor: '#0f172a',
          border: '1px solid #334155',
          borderRadius: '8px',
          padding: '12px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.8125rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <HelpCircle size={16} style={{ color: '#06b6d4' }} />
          <span style={{ color: '#cbd5e1' }}>
            Attribution Engine: <strong>GNNExplainer (Module 5)</strong> evaluated against GAT Attention & Random Baselines.
          </span>
        </div>
        <div style={{ color: '#94a3b8' }}>
          Fraud Probability: <strong style={{ color: '#f87171' }}>{(explanation.prediction_probability * 100).toFixed(2)}%</strong>
        </div>
      </div>

      {/* Quantitative Faithfulness & Sparsity Cards */}
      <FidelityCard explanation={explanation} />

      {/* Feature & Edge Tables */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(440px, 1fr))', gap: '20px' }}>
        <FeatureAttributionTable features={explanation.top_features} />
        <EdgeAttributionTable edges={explanation.top_edges} />
      </div>
    </div>
  );
};
