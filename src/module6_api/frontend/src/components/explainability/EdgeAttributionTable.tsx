import React from 'react';
import { EdgeAttributionItem } from '../../types/api';

interface EdgeAttributionTableProps {
  edges: EdgeAttributionItem[];
}

export const EdgeAttributionTable: React.FC<EdgeAttributionTableProps> = ({ edges }) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">Top Influential Payment Flow Edges</div>
        <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
          GNNExplainer continuous edge mask attribution (Ranked 1–{edges.length})
        </div>
      </div>

      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Source Tx ID</th>
              <th>Target Tx ID</th>
              <th>Edge Importance</th>
              <th style={{ width: '160px' }}>Relative Weight</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((edge) => {
              const normPct = (edge.normalized_importance * 100).toFixed(1);
              return (
                <tr key={`${edge.source_tx_id}-${edge.target_tx_id}`}>
                  <td style={{ fontWeight: 600, color: '#64748b' }}>#{edge.rank}</td>
                  <td className="font-mono" style={{ color: '#38bdf8' }}>
                    {edge.source_tx_id.slice(0, 10)}...
                  </td>
                  <td className="font-mono" style={{ color: '#cbd5e1' }}>
                    {edge.target_tx_id.slice(0, 10)}...
                  </td>
                  <td style={{ fontWeight: 600 }}>{edge.raw_importance.toFixed(4)}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          flex: 1,
                          height: '6px',
                          backgroundColor: '#1e293b',
                          borderRadius: '3px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${normPct}%`,
                            height: '100%',
                            backgroundColor: '#f59e0b',
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', width: '36px' }}>{normPct}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
