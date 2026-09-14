import React from 'react';
import { Network, HelpCircle, ShieldAlert, ChevronLeft, ChevronRight } from 'lucide-react';
import { TransactionResponse } from '../../types/api';
import { Badge } from '../common/Badge';

interface TransactionTableProps {
  transactions: TransactionResponse[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  onPageChange: (newPage: number) => void;
  onSelectTx: (txId: string) => void;
  onNavigateTab: (tab: any, txId: string) => void;
  onOpenReviewModal?: (tx: TransactionResponse) => void;
}

export const TransactionTable: React.FC<TransactionTableProps> = ({
  transactions,
  total,
  page,
  pageSize,
  totalPages,
  onPageChange,
  onSelectTx,
  onNavigateTab,
  onOpenReviewModal,
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Timestep</th>
              <th>Predicted Fraud Prob</th>
              <th>Risk Tier</th>
              <th>Uncertainty</th>
              <th>Triage Priority</th>
              <th>Review Status</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((tx) => {
              const prob = tx.predicted_prob !== undefined ? tx.predicted_prob : 0.0;
              const probPercent = (prob * 100).toFixed(1);
              return (
                <tr key={tx.tx_id}>
                  <td
                    className="font-mono"
                    style={{ color: '#38bdf8', fontWeight: 600, cursor: 'pointer' }}
                    onClick={() => onSelectTx(tx.tx_id)}
                  >
                    {tx.tx_id}
                  </td>
                  <td>
                    <span style={{ color: '#94a3b8' }}>TS #{tx.timestep}</span>
                  </td>
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
                            width: `${probPercent}%`,
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
                    <Badge type="risk" value={tx.risk_level || 'LOW'} />
                  </td>
                  <td>
                    <span style={{ color: '#cbd5e1' }}>
                      {tx.uncertainty_score !== undefined ? tx.uncertainty_score.toFixed(3) : '-'}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontWeight: 600, color: '#f8fafc' }}>
                      {tx.priority_score !== undefined ? `${(tx.priority_score * 100).toFixed(1)}%` : '-'}
                    </span>
                  </td>
                  <td>
                    <Badge type="status" value={tx.triage_status} />
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <div style={{ display: 'inline-flex', gap: '6px' }}>
                      <button
                        onClick={() => onNavigateTab('graph', tx.tx_id)}
                        className="btn btn-secondary"
                        style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                        title="View 2-Hop Graph"
                      >
                        <Network size={13} /> Graph
                      </button>
                      <button
                        onClick={() => onNavigateTab('explain', tx.tx_id)}
                        className="btn btn-secondary"
                        style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                        title="Explain Attribution"
                      >
                        <HelpCircle size={13} /> XAI
                      </button>
                      {onOpenReviewModal && (
                        <button
                          onClick={() => onOpenReviewModal(tx)}
                          className="btn btn-primary"
                          style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                          title="Submit Review Verdict"
                        >
                          <ShieldAlert size={13} /> Review
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 4px',
          fontSize: '0.8125rem',
          color: '#94a3b8',
        }}
      >
        <div>
          Showing {Math.min((page - 1) * pageSize + 1, total)} to{' '}
          {Math.min(page * pageSize, total)} of {total.toLocaleString()} transactions
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page <= 1}
            className="btn btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.75rem' }}
          >
            <ChevronLeft size={14} /> Prev
          </button>
          <span>
            Page <strong style={{ color: '#f8fafc' }}>{page}</strong> of{' '}
            <strong style={{ color: '#f8fafc' }}>{totalPages}</strong>
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages}
            className="btn btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.75rem' }}
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};
