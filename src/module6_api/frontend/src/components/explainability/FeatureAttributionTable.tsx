import React from 'react';
import { FeatureAttributionItem } from '../../types/api';

interface FeatureAttributionTableProps {
  features: FeatureAttributionItem[];
}

export const FeatureAttributionTable: React.FC<FeatureAttributionTableProps> = ({ features }) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">Top Attributed Transaction Features</div>
        <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
          GNNExplainer feature mask attribution (Ranked 1–{features.length})
        </div>
      </div>

      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Feature Name</th>
              <th>Type</th>
              <th>Importance Score</th>
              <th style={{ width: '160px' }}>Relative Weight</th>
            </tr>
          </thead>
          <tbody>
            {features.map((feat) => {
              const normPct = (feat.normalized_importance * 100).toFixed(1);
              return (
                <tr key={feat.feature_index}>
                  <td style={{ fontWeight: 600, color: '#64748b' }}>#{feat.rank}</td>
                  <td className="font-mono" style={{ color: '#f8fafc', fontWeight: 500 }}>
                    {feat.feature_name}
                  </td>
                  <td>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        backgroundColor: feat.feature_type === 'local' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(168, 85, 247, 0.15)',
                        color: feat.feature_type === 'local' ? '#38bdf8' : '#c084fc',
                        border: `1px solid ${feat.feature_type === 'local' ? '#0284c7' : '#9333ea'}`,
                      }}
                    >
                      {feat.feature_type}
                    </span>
                  </td>
                  <td style={{ fontWeight: 600 }}>{feat.raw_importance.toFixed(4)}</td>
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
                            backgroundColor: '#06b6d4',
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
