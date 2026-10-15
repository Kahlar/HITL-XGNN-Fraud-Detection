import React, { useEffect, useState } from 'react';
import {
  Network,
  HelpCircle,
  ShieldAlert,
  X,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Info,
  BarChart2,
  Cpu,
} from 'lucide-react';
import { getGroupEvaluationMetrics, getTransactionDetail } from '../../api/client';
import { GroupEvaluationResponse, TransactionDetailResponse, TransactionResponse } from '../../types/api';
import { Badge } from '../common/Badge';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface PredictionDetailPanelProps {
  tx: TransactionResponse;
  currentSearchIds?: string[];
  onClose: () => void;
  onNavigateTab: (tab: any, txId: string) => void;
  onOpenReviewModal?: (tx: TransactionResponse) => void;
}

export const PredictionDetailPanel: React.FC<PredictionDetailPanelProps> = ({
  tx,
  currentSearchIds,
  onClose,
  onNavigateTab,
  onOpenReviewModal,
}) => {
  const [detail, setDetail] = useState<TransactionDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(true);
  const [evalScope, setEvalScope] = useState<'timestep' | 'filtered' | 'reviewed'>('timestep');
  const [groupEval, setGroupEval] = useState<GroupEvaluationResponse | null>(null);
  const [loadingEval, setLoadingEval] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  // Fetch full transaction details
  useEffect(() => {
    let isMounted = true;
    setLoadingDetail(true);
    getTransactionDetail(tx.tx_id)
      .then((data) => {
        if (isMounted) {
          setDetail(data);
          setLoadingDetail(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setLoadingDetail(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [tx.tx_id]);

  // Fetch group evaluation whenever scope changes
  useEffect(() => {
    let isMounted = true;
    setLoadingEval(true);
    setEvalError(null);

    const params: any = { scope: evalScope };
    if (evalScope === 'timestep') {
      params.timestep = tx.timestep;
    } else if (evalScope === 'filtered') {
      params.timestep = tx.timestep;
      params.tx_ids = currentSearchIds && currentSearchIds.length > 0 ? currentSearchIds : [tx.tx_id];
    }

    getGroupEvaluationMetrics(params)
      .then((res) => {
        if (isMounted) {
          setGroupEval(res);
          setLoadingEval(false);
        }
      })
      .catch((err: any) => {
        if (isMounted) {
          setEvalError(err.message || 'Failed to compute group evaluation metrics');
          setLoadingEval(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [evalScope, tx.tx_id, tx.timestep, currentSearchIds]);

  const prob = detail?.predicted_prob ?? tx.predicted_prob ?? 0;
  const probPercent = (prob * 100).toFixed(1);
  const threshold = detail?.decision_threshold ?? 0.5517;
  const predClass = detail?.predicted_class ?? (prob >= threshold ? 1 : 0);
  const gtLabel = detail?.ground_truth_label ?? tx.ground_truth_label;
  const outcome = detail?.outcome_category ?? (
    gtLabel === -1
      ? 'NOT_EVALUABLE'
      : predClass === 1 && gtLabel === 1
      ? 'TRUE_POSITIVE'
      : predClass === 1 && gtLabel === 0
      ? 'FALSE_POSITIVE'
      : predClass === 0 && gtLabel === 0
      ? 'TRUE_NEGATIVE'
      : 'FALSE_NEGATIVE'
  );

  const uncertainty = detail?.uncertainty_score ?? tx.uncertainty_score ?? (1 - 2 * Math.abs(prob - 0.5));
  const entropy = detail?.entropy ?? tx.entropy ?? 0;

  return (
    <div
      className="card"
      style={{
        border: '1px solid #38bdf8',
        boxShadow: '0 4px 20px rgba(56, 189, 248, 0.12)',
        backgroundColor: '#0f172a',
        marginBottom: '20px',
      }}
    >
      {/* Header bar */}
      <div
        className="card-header"
        style={{
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '12px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Cpu size={20} style={{ color: '#38bdf8' }} />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                Prediction & Decision Details
              </span>
              <span className="font-mono" style={{ color: '#38bdf8', fontWeight: 700, fontSize: '0.95rem' }}>
                #{tx.tx_id}
              </span>
              <span
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  backgroundColor: '#1e293b',
                  color: '#94a3b8',
                  padding: '2px 8px',
                  borderRadius: '12px',
                }}
              >
                Timestep {tx.timestep}
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => onNavigateTab('graph', tx.tx_id)}
            className="btn btn-secondary"
            style={{ padding: '5px 10px', fontSize: '0.8rem' }}
          >
            <Network size={14} /> 2-Hop Graph
          </button>
          <button
            onClick={() => onNavigateTab('explain', tx.tx_id)}
            className="btn btn-secondary"
            style={{ padding: '5px 10px', fontSize: '0.8rem' }}
          >
            <HelpCircle size={14} /> Explain (XAI)
          </button>
          {onOpenReviewModal && (
            <button
              onClick={() => onOpenReviewModal(tx)}
              className="btn btn-primary"
              style={{ padding: '5px 10px', fontSize: '0.8rem' }}
            >
              <ShieldAlert size={14} /> Review
            </button>
          )}
          <button
            onClick={onClose}
            className="btn btn-secondary"
            style={{ padding: '5px 8px', color: '#94a3b8' }}
            title="Close Panel"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {loadingDetail ? (
        <div style={{ padding: '24px' }}>
          <LoadingSpinner message="Loading live model inference telemetry..." />
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', padding: '16px' }}>
          {/* Section 1: Transaction-Level Model Telemetry Grid */}
          <div>
            <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
              Transaction-Specific Prediction Telemetry (Feature A)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
              {/* Active Model */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>ACTIVE MODEL</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>
                  {detail?.model_version || 'graphsage_hitl_uncertainty_20'}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#38bdf8' }}>Architecture: GraphSAGE (2-layer Mean)</div>
              </div>

              {/* Fraud Probability */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>FRAUD PROBABILITY (p)</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: prob >= 0.5517 ? '#f87171' : '#34d399' }}>
                  {prob.toFixed(4)} <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>({probPercent}%)</span>
                </div>
                <div style={{ width: '100%', height: '5px', backgroundColor: '#0f172a', borderRadius: '3px', marginTop: '6px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${probPercent}%`,
                      height: '100%',
                      backgroundColor: prob >= 0.85 ? '#ef4444' : prob >= 0.5517 ? '#f97316' : '#10b981',
                    }}
                  />
                </div>
              </div>

              {/* Threshold & Binary Decision */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>DECISION THRESHOLD & CLASS</div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: predClass === 1 ? '#f87171' : '#34d399' }}>
                    {predClass === 1 ? 'Illicit (1)' : 'Licit (0)'}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#fbbf24' }}>τ* = {threshold.toFixed(4)}</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>
                  Rule: {prob >= threshold ? `p ≥ ${threshold.toFixed(4)}` : `p < ${threshold.toFixed(4)}`}
                </div>
              </div>

              {/* Ground Truth Label */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>GROUND TRUTH LABEL</div>
                <div>
                  {gtLabel === 1 ? (
                    <span style={{ color: '#f87171', fontWeight: 700, fontSize: '0.95rem' }}>Illicit (Class 1)</span>
                  ) : gtLabel === 0 ? (
                    <span style={{ color: '#34d399', fontWeight: 700, fontSize: '0.95rem' }}>Licit (Class 0)</span>
                  ) : (
                    <span style={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.8rem' }}>
                      Unavailable (Unlabelled / Test)
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                  Raw Class: {gtLabel === -1 ? 'unknown (-1)' : String(gtLabel)}
                </div>
              </div>

              {/* Individual Outcome Category */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>INDIVIDUAL OUTCOME CATEGORY</div>
                <div>
                  {outcome === 'TRUE_POSITIVE' && (
                    <span style={{ color: '#34d399', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <CheckCircle2 size={15} /> True Positive (TP)
                    </span>
                  )}
                  {outcome === 'TRUE_NEGATIVE' && (
                    <span style={{ color: '#34d399', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <CheckCircle2 size={15} /> True Negative (TN)
                    </span>
                  )}
                  {outcome === 'FALSE_POSITIVE' && (
                    <span style={{ color: '#f97316', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <AlertTriangle size={15} /> False Positive (FP)
                    </span>
                  )}
                  {outcome === 'FALSE_NEGATIVE' && (
                    <span style={{ color: '#ef4444', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <AlertTriangle size={15} /> False Negative (FN)
                    </span>
                  )}
                  {outcome === 'NOT_EVALUABLE' && (
                    <span style={{ color: '#94a3b8', fontWeight: 600, fontSize: '0.8rem' }}>
                      Not evaluable (Label unavailable)
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '4px' }}>
                  Single transaction outcome (not aggregate metric)
                </div>
              </div>

              {/* Uncertainty & Entropy */}
              <div style={{ padding: '12px', backgroundColor: '#162032', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '4px' }}>UNCERTAINTY & ENTROPY</div>
                <div style={{ fontSize: '0.9rem', color: '#f8fafc', fontWeight: 600 }}>
                  Margin: <span style={{ color: '#38bdf8' }}>{uncertainty.toFixed(4)}</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                  Entropy H(p): <span style={{ color: '#cbd5e1' }}>{entropy.toFixed(4)}</span>
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
                  U(p) = 1 - 2·|p - 0.5|
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Group-Level F1 & PR-AUC Evaluation (Feature B) */}
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <div>
                <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#f8fafc' }}>
                  Group-Level Evaluation Metrics (Feature B)
                </span>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', marginLeft: '10px' }}>
                  Aggregate F1 & PR-AUC across selected cohort (unlabelled nodes masked)
                </span>
              </div>

              {/* Scope Selector */}
              <div style={{ display: 'flex', gap: '6px', backgroundColor: '#162032', padding: '3px', borderRadius: '6px' }}>
                <button
                  onClick={() => setEvalScope('timestep')}
                  className={`btn ${evalScope === 'timestep' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                >
                  Timestep #{tx.timestep}
                </button>
                <button
                  onClick={() => setEvalScope('filtered')}
                  className={`btn ${evalScope === 'filtered' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                >
                  Search Subset
                </button>
                <button
                  onClick={() => setEvalScope('reviewed')}
                  className={`btn ${evalScope === 'reviewed' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                >
                  Reviewed Batch
                </button>
              </div>
            </div>

            {loadingEval ? (
              <LoadingSpinner message={`Evaluating ${evalScope} cohort metrics...`} size="sm" />
            ) : evalError ? (
              <div style={{ color: '#f87171', fontSize: '0.8rem', padding: '8px', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px' }}>
                {evalError}
              </div>
            ) : groupEval ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {/* Metrics Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px' }}>
                  <div style={{ padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>TOTAL TRANSACTIONS</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                      {(groupEval.total_transactions ?? 0).toLocaleString()}
                    </div>
                  </div>

                  <div style={{ padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>LABELED NODES</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>
                      {(groupEval.labeled_count ?? 0).toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
                      {(groupEval.unlabeled_count ?? 0).toLocaleString()} masked
                    </div>
                  </div>

                  <div style={{ padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>ILLICIT F1 SCORE</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#34d399' }}>
                      {groupEval.f1_score != null ? groupEval.f1_score.toFixed(4) : '-'}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#fbbf24' }}>
                      @ τ*={(groupEval.decision_threshold ?? 0.5517).toFixed(4)}
                    </div>
                  </div>

                  <div style={{ padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>PR-AUC (CURVE)</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>
                      {groupEval.pr_auc != null ? groupEval.pr_auc.toFixed(4) : '-'}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>Trapezoidal AUC</div>
                  </div>

                  <div style={{ padding: '10px', backgroundColor: '#1e293b', borderRadius: '6px' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>PRECISION / RECALL</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>
                      {groupEval.precision != null ? `${(groupEval.precision * 100).toFixed(1)}%` : '-'} /{' '}
                      {groupEval.recall != null ? `${(groupEval.recall * 100).toFixed(1)}%` : '-'}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>P / R</div>
                  </div>
                </div>

                {/* 2x2 Confusion Matrix Visual */}
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center', backgroundColor: '#162032', padding: '12px', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', minWidth: '110px' }}>
                    CONFUSION MATRIX:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', flex: 1, maxWidth: '400px' }}>
                    <div style={{ padding: '6px 10px', backgroundColor: '#1e293b', borderRadius: '4px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.65rem', color: '#64748b' }}>TRUE POSITIVES (TP)</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: '#34d399' }}>{groupEval.true_positives ?? 0}</div>
                    </div>
                    <div style={{ padding: '6px 10px', backgroundColor: '#1e293b', borderRadius: '4px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.65rem', color: '#64748b' }}>FALSE POSITIVES (FP)</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f97316' }}>{groupEval.false_positives ?? 0}</div>
                    </div>
                    <div style={{ padding: '6px 10px', backgroundColor: '#1e293b', borderRadius: '4px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.65rem', color: '#64748b' }}>FALSE NEGATIVES (FN)</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ef4444' }}>{groupEval.false_negatives ?? 0}</div>
                    </div>
                    <div style={{ padding: '6px 10px', backgroundColor: '#1e293b', borderRadius: '4px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.65rem', color: '#64748b' }}>TRUE NEGATIVES (TN)</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700, color: '#34d399' }}>{groupEval.true_negatives ?? 0}</div>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', flex: 1 }}>
                    <Info size={14} style={{ display: 'inline', marginRight: '4px', color: '#38bdf8' }} />
                    {groupEval.notes || 'Evaluated strictly over labeled nodes.'}
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          {/* Section 3: Threshold & Decision Policy Explanation (Feature C) */}
          <div
            style={{
              borderTop: '1px solid var(--border-color)',
              paddingTop: '12px',
              fontSize: '0.8rem',
              color: '#94a3b8',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <span style={{ fontWeight: 600, color: '#fbbf24' }}>
                <Sliders size={13} style={{ display: 'inline', marginRight: '4px' }} />
                Threshold Calibration Policy (Feature C):
              </span>{' '}
              minority Illicit F1 maximized over 200 grid points in [0.01, 0.99] strictly on validation split ($t \in [35, 39]$).
              Active GNN threshold is <strong style={{ color: '#f8fafc' }}>τ* = 0.5517</strong> (baseline XGBoost τ* = 0.9014).
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
