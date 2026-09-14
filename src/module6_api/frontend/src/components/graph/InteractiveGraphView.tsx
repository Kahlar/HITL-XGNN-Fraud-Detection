import React, { useEffect, useState } from 'react';
import { getTransactionSubgraph } from '../../api/client';
import { GraphNode, SubgraphResponse } from '../../types/api';
import { GraphCanvas } from './GraphCanvas';
import { GraphControls } from './GraphControls';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorAlert } from '../common/ErrorAlert';
import { Search, Network, HelpCircle } from 'lucide-react';

interface InteractiveGraphViewProps {
  initialTxId?: string;
  onNavigateTab: (tab: any, txId: string) => void;
}

export const InteractiveGraphView: React.FC<InteractiveGraphViewProps> = ({
  initialTxId = '230425980',
  onNavigateTab,
}) => {
  const [txId, setTxId] = useState(initialTxId);
  const [inputTxId, setInputTxId] = useState(initialTxId);
  const [hops, setHops] = useState(2);
  const [subgraph, setSubgraph] = useState<SubgraphResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSubgraphData = async (targetId: string, currentHops: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTransactionSubgraph(targetId, currentHops);
      setSubgraph(data);
      const target = data.nodes.find((n) => n.is_target) || data.nodes[0] || null;
      setSelectedNode(target);
    } catch (err: any) {
      setError(err.message || `Failed to extract computational subgraph for ${targetId}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (txId) {
      fetchSubgraphData(txId, hops);
    }
  }, [txId, hops]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputTxId.trim() && inputTxId.trim() !== txId) {
      setTxId(inputTxId.trim());
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header & Search Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
            Interactive 2-Hop Computational Graph Canvas
          </h2>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            Inspect multi-hop payment chains and message-passing neighborhoods extracted via PyTorch Geometric.
          </p>
        </div>

        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '8px' }}>
          <div style={{ position: 'relative' }}>
            <Search size={15} style={{ position: 'absolute', left: '10px', top: '10px', color: '#64748b' }} />
            <input
              type="text"
              placeholder="Target Tx ID..."
              value={inputTxId}
              onChange={(e) => setInputTxId(e.target.value)}
              className="font-mono"
              style={{
                backgroundColor: '#1e293b',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '8px 12px 8px 32px',
                color: '#f8fafc',
                fontSize: '0.8125rem',
                width: '200px',
              }}
            />
          </div>
          <button type="submit" className="btn btn-secondary" style={{ padding: '8px 14px' }}>
            Render
          </button>
        </form>
      </div>

      {/* Graph Toolbar Controls */}
      {subgraph && (
        <GraphControls
          hops={hops}
          onHopsChange={(newHops) => setHops(newHops)}
          onZoomIn={() => {}}
          onZoomOut={() => {}}
          onResetZoom={() => {}}
          numNodes={subgraph.num_nodes}
          numEdges={subgraph.num_edges}
        />
      )}

      {/* Canvas View */}
      {loading ? (
        <LoadingSpinner message={`Extracting ${hops}-hop induced neighborhood for ${txId}...`} size="lg" />
      ) : error ? (
        <ErrorAlert message={error} onRetry={() => fetchSubgraphData(txId, hops)} />
      ) : subgraph ? (
        <GraphCanvas
          subgraph={subgraph}
          selectedNode={selectedNode}
          onSelectNode={(node) => setSelectedNode(node)}
          onInvestigateNode={(newTx) => {
            setTxId(newTx);
            setInputTxId(newTx);
          }}
          onExplainNode={(newTx) => onNavigateTab('explain', newTx)}
        />
      ) : null}
    </div>
  );
};
