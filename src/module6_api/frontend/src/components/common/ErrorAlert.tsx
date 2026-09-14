import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorAlertProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = 'API Error Occurred',
  message,
  onRetry,
}) => {
  return (
    <div
      style={{
        backgroundColor: 'rgba(185, 28, 28, 0.15)',
        border: '1px solid #ef4444',
        borderRadius: '8px',
        padding: '16px',
        margin: '16px 0',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
      }}
    >
      <AlertTriangle size={20} style={{ color: '#ef4444', flexShrink: 0, marginTop: '2px' }} />
      <div style={{ flex: 1 }}>
        <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f87171', marginBottom: '4px' }}>
          {title}
        </h4>
        <p style={{ fontSize: '0.8125rem', color: '#cbd5e1' }}>{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="btn btn-secondary"
          style={{ padding: '6px 12px', fontSize: '0.75rem', flexShrink: 0 }}
        >
          <RefreshCw size={14} /> Retry
        </button>
      )}
    </div>
  );
};
