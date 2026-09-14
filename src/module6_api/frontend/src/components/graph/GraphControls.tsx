import React from 'react';
import { ZoomIn, ZoomOut, Maximize2, Layers } from 'lucide-react';

interface GraphControlsProps {
  hops: number;
  onHopsChange: (hops: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  numNodes: number;
  numEdges: number;
}

export const GraphControls: React.FC<GraphControlsProps> = ({
  hops,
  onHopsChange,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  numNodes,
  numEdges,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: '#0c1222',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '10px 16px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem' }}>
          <Layers size={16} style={{ color: '#06b6d4' }} />
          <span style={{ color: '#94a3b8' }}>Neighborhood Hops:</span>
          <div style={{ display: 'flex', gap: '4px' }}>
            {[1, 2, 3].map((h) => (
              <button
                key={h}
                onClick={() => onHopsChange(h)}
                style={{
                  padding: '3px 8px',
                  borderRadius: '4px',
                  border: '1px solid',
                  borderColor: hops === h ? '#06b6d4' : '#334155',
                  backgroundColor: hops === h ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                  color: hops === h ? '#38bdf8' : '#94a3b8',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {h}-Hop
              </button>
            ))}
          </div>
        </div>

        <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
          Graph Subgraph: <strong style={{ color: '#f8fafc' }}>{numNodes}</strong> nodes,{' '}
          <strong style={{ color: '#f8fafc' }}>{numEdges}</strong> directed payment edges
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <button
          onClick={onZoomIn}
          className="btn btn-secondary"
          style={{ padding: '6px', fontSize: '0.75rem' }}
          title="Zoom In"
        >
          <ZoomIn size={15} />
        </button>
        <button
          onClick={onZoomOut}
          className="btn btn-secondary"
          style={{ padding: '6px', fontSize: '0.75rem' }}
          title="Zoom Out"
        >
          <ZoomOut size={15} />
        </button>
        <button
          onClick={onResetZoom}
          className="btn btn-secondary"
          style={{ padding: '6px', fontSize: '0.75rem' }}
          title="Reset Zoom"
        >
          <Maximize2 size={15} />
        </button>
      </div>
    </div>
  );
};
