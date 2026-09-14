import React from 'react';
import { CheckCircle, AlertCircle, Gauge, Zap } from 'lucide-react';
import { ExplanationResponse } from '../../types/api';

interface FidelityCardProps {
  explanation: ExplanationResponse;
}

export const FidelityCard: React.FC<FidelityCardProps> = ({ explanation }) => {
  const fidPlus = explanation.fidelity_plus ?? 0.0;
  const fidMinus = explanation.fidelity_minus ?? 0.0;
  const edgeSparsity = (explanation.edge_sparsity ?? 0.0) * 100;
  const featSparsity = (explanation.feature_sparsity ?? 0.0) * 100;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
      {/* Fidelity+ Card */}
      <div className="card" style={{ padding: '14px 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '6px' }}>
          <span>Fidelity+ (Sufficiency)</span>
          <CheckCircle size={15} style={{ color: '#34d399' }} />
        </div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: fidPlus > 0 ? '#34d399' : '#f87171' }}>
          {fidPlus >= 0 ? `+${fidPlus.toFixed(4)}` : fidPlus.toFixed(4)}
        </div>
        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
          Prediction drop when removing explanation subgraph
        </div>
      </div>

      {/* Fidelity- Card */}
      <div className="card" style={{ padding: '14px 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '6px' }}>
          <span>Fidelity- (Necessity)</span>
          <AlertCircle size={15} style={{ color: '#38bdf8' }} />
        </div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8' }}>
          {fidMinus >= 0 ? `+${fidMinus.toFixed(4)}` : fidMinus.toFixed(4)}
        </div>
        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
          Deviation when retaining ONLY explanation subgraph
        </div>
      </div>

      {/* Sparsity Card */}
      <div className="card" style={{ padding: '14px 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '6px' }}>
          <span>Graph & Feature Sparsity</span>
          <Gauge size={15} style={{ color: '#fbbf24' }} />
        </div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fbbf24' }}>
          {edgeSparsity.toFixed(1)}% <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>/ {featSparsity.toFixed(1)}%</span>
        </div>
        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
          Edge mask sparsity / Feature mask sparsity
        </div>
      </div>

      {/* Latency Card */}
      <div className="card" style={{ padding: '14px 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '6px' }}>
          <span>Attribution Latency</span>
          <Zap size={15} style={{ color: '#06b6d4' }} />
        </div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#06b6d4' }}>
          {explanation.generation_latency_ms.toFixed(1)} ms
        </div>
        <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
          GNNExplainer optimization time (60 epochs)
        </div>
      </div>
    </div>
  );
};
