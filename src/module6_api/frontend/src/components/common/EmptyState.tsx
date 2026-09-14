import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No records found',
  description = 'Try adjusting your search criteria or filters to find results.',
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 24px',
        textAlign: 'center',
        border: '1px dashed var(--border-color)',
        borderRadius: '8px',
        backgroundColor: 'rgba(15, 23, 42, 0.4)',
      }}
    >
      <Inbox size={36} style={{ color: '#64748b', marginBottom: '12px' }} />
      <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '6px' }}>
        {title}
      </h3>
      <p style={{ fontSize: '0.875rem', color: '#94a3b8', maxWidth: '400px' }}>
        {description}
      </p>
    </div>
  );
};
