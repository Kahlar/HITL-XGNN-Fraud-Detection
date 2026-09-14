import React from 'react';
import { Search, Filter, RotateCcw } from 'lucide-react';
import { TransactionFilterParams } from '../../api/client';

interface TransactionFiltersProps {
  filters: TransactionFilterParams;
  searchId: string;
  onSearchIdChange: (val: string) => void;
  onFilterChange: (newFilters: Partial<TransactionFilterParams>) => void;
  onReset: () => void;
}

export const TransactionFilters: React.FC<TransactionFiltersProps> = ({
  filters,
  searchId,
  onSearchIdChange,
  onFilterChange,
  onReset,
}) => {
  return (
    <div
      style={{
        backgroundColor: '#0f172a',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '16px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '12px',
        alignItems: 'center',
      }}
    >
      {/* Search by Tx ID */}
      <div style={{ flex: '1 1 200px', position: 'relative' }}>
        <Search
          size={16}
          style={{ position: 'absolute', left: '10px', top: '10px', color: '#64748b' }}
        />
        <input
          type="text"
          placeholder="Search Transaction ID..."
          value={searchId}
          onChange={(e) => onSearchIdChange(e.target.value)}
          style={{
            width: '100%',
            backgroundColor: '#1e293b',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            padding: '8px 12px 8px 32px',
            color: '#f8fafc',
            fontSize: '0.875rem',
            outline: 'none',
          }}
        />
      </div>

      {/* Timestep Filter */}
      <div style={{ minWidth: '140px' }}>
        <select
          value={filters.timestep || ''}
          onChange={(e) =>
            onFilterChange({
              timestep: e.target.value ? parseInt(e.target.value, 10) : undefined,
              page: 1,
            })
          }
          style={{
            width: '100%',
            backgroundColor: '#1e293b',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            padding: '8px 12px',
            color: '#f8fafc',
            fontSize: '0.875rem',
            outline: 'none',
          }}
        >
          <option value="">All Timesteps (1-49)</option>
          {Array.from({ length: 49 }, (_, i) => i + 1).map((ts) => (
            <option key={ts} value={ts}>
              Timestep {ts} {ts >= 40 ? '(Test)' : ts >= 35 ? '(Val)' : '(Train)'}
            </option>
          ))}
        </select>
      </div>

      {/* Risk Level Filter */}
      <div style={{ minWidth: '140px' }}>
        <select
          value={filters.risk_level || ''}
          onChange={(e) => onFilterChange({ risk_level: e.target.value || undefined, page: 1 })}
          style={{
            width: '100%',
            backgroundColor: '#1e293b',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            padding: '8px 12px',
            color: '#f8fafc',
            fontSize: '0.875rem',
            outline: 'none',
          }}
        >
          <option value="">All Risk Tiers</option>
          <option value="CRITICAL">Critical Risk (p ≥ 0.85)</option>
          <option value="HIGH">High Risk (p ≥ 0.70)</option>
          <option value="MEDIUM">Medium Risk (p ≥ 0.40)</option>
          <option value="LOW">Low Risk (p &lt; 0.40)</option>
        </select>
      </div>

      {/* Triage Status Filter */}
      <div style={{ minWidth: '140px' }}>
        <select
          value={filters.triage_status || ''}
          onChange={(e) => onFilterChange({ triage_status: e.target.value || undefined, page: 1 })}
          style={{
            width: '100%',
            backgroundColor: '#1e293b',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            padding: '8px 12px',
            color: '#f8fafc',
            fontSize: '0.875rem',
            outline: 'none',
          }}
        >
          <option value="">All Review Statuses</option>
          <option value="QUEUED">QUEUED for Review</option>
          <option value="REVIEWED">REVIEWED by Analyst</option>
          <option value="ESCALATED">ESCALATED for Investigation</option>
          <option value="PENDING">PENDING</option>
        </select>
      </div>

      {/* Reset Button */}
      <button
        onClick={onReset}
        className="btn btn-secondary"
        style={{ padding: '8px 12px' }}
        title="Reset filters"
      >
        <RotateCcw size={14} /> Reset
      </button>
    </div>
  );
};
