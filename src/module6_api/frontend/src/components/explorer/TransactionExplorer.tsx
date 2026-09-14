import React, { useEffect, useState } from 'react';
import { getTransactions, TransactionFilterParams } from '../../api/client';
import { TransactionResponse } from '../../types/api';
import { TransactionFilters } from './TransactionFilters';
import { TransactionTable } from './TransactionTable';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorAlert } from '../common/ErrorAlert';
import { EmptyState } from '../common/EmptyState';

interface TransactionExplorerProps {
  onSelectTx: (txId: string) => void;
  onNavigateTab: (tab: any, txId: string) => void;
  onOpenReviewModal?: (tx: TransactionResponse) => void;
}

export const TransactionExplorer: React.FC<TransactionExplorerProps> = ({
  onSelectTx,
  onNavigateTab,
  onOpenReviewModal,
}) => {
  const [filters, setFilters] = useState<TransactionFilterParams>({
    page: 1,
    page_size: 25,
  });
  const [searchId, setSearchId] = useState('');
  const [transactions, setTransactions] = useState<TransactionResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Debounce search input and sync to filters
  useEffect(() => {
    const handler = setTimeout(() => {
      const trimmed = searchId.trim();
      setFilters((prev) => {
        const nextSearch = trimmed || undefined;
        if (prev.search === nextSearch) return prev;
        return {
          ...prev,
          search: nextSearch,
          page: 1,
        };
      });
    }, 300);

    return () => clearTimeout(handler);
  }, [searchId]);

  const fetchTxList = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getTransactions(filters);
      setTransactions(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages || 1);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch transactions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTxList();
  }, [filters]);

  const handleFilterChange = (newFilters: Partial<TransactionFilterParams>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  const handleReset = () => {
    setSearchId('');
    setFilters({ page: 1, page_size: 25 });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
          Transaction Surveillance Explorer
        </h2>
        <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
          Search, filter, and inspect Bitcoin transactions across all 49 discrete temporal graph snapshots.
        </p>
      </div>

      <TransactionFilters
        filters={filters}
        searchId={searchId}
        onSearchIdChange={setSearchId}
        onFilterChange={handleFilterChange}
        onReset={handleReset}
      />

      {loading ? (
        <LoadingSpinner message="Querying transaction graph..." />
      ) : error ? (
        <ErrorAlert message={error} onRetry={fetchTxList} />
      ) : transactions.length === 0 ? (
        <EmptyState title="No transactions match the selected filters" />
      ) : (
        <TransactionTable
          transactions={transactions}
          total={total}
          page={filters.page || 1}
          pageSize={filters.page_size || 25}
          totalPages={totalPages}
          onPageChange={(newPage) => handleFilterChange({ page: newPage })}
          onSelectTx={onSelectTx}
          onNavigateTab={onNavigateTab}
          onOpenReviewModal={onOpenReviewModal}
        />
      )}
    </div>
  );
};
