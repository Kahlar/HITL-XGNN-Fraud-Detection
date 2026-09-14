import React from 'react';
import { Network, HelpCircle, ShieldAlert } from 'lucide-react';
import { TriageQueueItem } from '../../types/api';
import { Badge } from '../common/Badge';

interface TriageQueueTableProps {
  items: TriageQueueItem[];
  onOpenReview: (item: TriageQueueItem) => void;
  onNavigateTab: (tab: any, txId: string) => void;
}

export const TriageQueueTable: React.FC<TriageQueueTableProps> = ({
  items,
  onOpenReview,
  onNavigateTab,
}) => {
  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Priority Rank</th>
            <th>Transaction ID</th>
            <th>Timestep</th>
            <th>Predicted Fraud Score</th>
            <th>Risk Tier</th>
            <th>Margin Uncertainty</th>
            <th>Entropy</th>
            <th>Composite Priority</th>
            <th style={{ textAlign: 'right' }}>Analyst Actions</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => {
            const prob = item.predicted_prob;
            const probPct = (prob * 100).toFixed(1);
            return (
              <tr key={item.tx_id}>
                <td style={{ fontWeight: 700, color: '#f59e0b' }}>#{idx + 1}</td>
                <td className="font-mono" style={{ color: '#38bdf8', fontWeight: 600 }}>
                  {item.tx_id}
                </td>
                <td>TS #{item.timestep}</td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div
                      style={{
                        width: '48px',
                        height: '6px',
                        backgroundColor: '#1e293b',
                        borderRadius: '3px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${probPct}%`,
                          height: '100%',
                          backgroundColor:
                            prob >= 0.85
                              ? '#ef4444'
                              : prob >= 0.70
                              ? '#f97316'
                              : prob >= 0.40
                              ? '#f59e0b'
                              : '#10b981',
                        }}
                      />
                    </div>
                    <span style={{ fontWeight: 600 }}>{prob.toFixed(4)}</span>
                  </div>
                </td>
                <td>
                  <Badge type="risk" value={item.risk_level} />
                </td>
                <td>
                  <span style={{ color: '#cbd5e1' }}>{item.uncertainty_score.toFixed(3)}</span>
                </td>
                <td>
                  <span style={{ color: '#cbd5e1' }}>{item.entropy.toFixed(3)} bits</span>
                </td>
                <td>
                  <span style={{ fontWeight: 700, color: '#fbbf24' }}>
                    {(item.priority_score * 100).toFixed(1)}%
                  </span>
                </td>
                <td style={{ textAlign: 'right' }}>
                  <div style={{ display: 'inline-flex', gap: '6px' }}>
                    <button
                      onClick={() => onNavigateTab('graph', item.tx_id)}
                      className="btn btn-secondary"
                      style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                      title="View Subgraph"
                    >
                      <Network size={13} /> Graph
                    </button>
                    <button
                      onClick={() => onNavigateTab('explain', item.tx_id)}
                      className="btn btn-secondary"
                      style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                      title="View Attributions"
                    >
                      <HelpCircle size={13} /> XAI
                    </button>
                    <button
                      onClick={() => onOpenReview(item)}
                      className="btn btn-primary"
                      style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                    >
                      <ShieldAlert size={13} /> Audit Case
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
