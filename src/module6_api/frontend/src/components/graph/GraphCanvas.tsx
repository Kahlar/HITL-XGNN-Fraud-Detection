import React, { useState, useMemo, useRef } from 'react';
import { SubgraphResponse, GraphNode, GraphEdge } from '../../types/api';
import { Badge } from '../common/Badge';
import { HelpCircle, ExternalLink } from 'lucide-react';

interface GraphCanvasProps {
  subgraph: SubgraphResponse;
  selectedNode: GraphNode | null;
  onSelectNode: (node: GraphNode) => void;
  onInvestigateNode: (txId: string) => void;
  onExplainNode: (txId: string) => void;
}

interface PositionedNode extends GraphNode {
  x: number;
  y: number;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  subgraph,
  selectedNode,
  onSelectNode,
  onInvestigateNode,
  onExplainNode,
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  const width = 800;
  const height = 540;
  const centerX = width / 2;
  const centerY = height / 2;

  // Calculate layout: Concentric radial layout centered on target transaction
  const positionedNodes = useMemo(() => {
    const nodes = subgraph.nodes;
    const target = nodes.find((n) => n.is_target) || nodes[0];
    const otherNodes = nodes.filter((n) => n.id !== target?.id);

    const positions: Record<string, PositionedNode> = {};

    if (target) {
      positions[target.id] = { ...target, x: centerX, y: centerY };
    }

    // Partition other nodes into concentric layers
    const totalOthers = otherNodes.length;
    const ring1Radius = Math.min(140, Math.max(80, totalOthers * 12));
    const ring2Radius = Math.min(240, Math.max(160, totalOthers * 18));

    otherNodes.forEach((node, idx) => {
      const isHop1 = idx % 2 === 0 || totalOthers < 8;
      const radius = isHop1 ? ring1Radius : ring2Radius;
      const angleStep = (2 * Math.PI) / totalOthers;
      const angle = idx * angleStep;

      positions[node.id] = {
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });

    return positions;
  }, [subgraph, centerX, centerY]);

  const getNodeColor = (node: GraphNode) => {
    if (node.risk_level === 'CRITICAL' || (node.predicted_prob != null && node.predicted_prob >= 0.85)) return '#ef4444';
    if (node.risk_level === 'HIGH' || (node.predicted_prob != null && node.predicted_prob >= 0.70)) return '#f97316';
    if (node.risk_level === 'MEDIUM' || (node.predicted_prob != null && node.predicted_prob >= 0.40)) return '#f59e0b';
    return '#10b981';
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.target === svgRef.current || (e.target as HTMLElement).tagName === 'svg') {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div style={{ display: 'flex', gap: '16px', position: 'relative' }}>
      {/* Interactive SVG Canvas */}
      <div
        style={{
          flex: 1,
          height: `${height}px`,
          backgroundColor: '#070b14',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          overflow: 'hidden',
          position: 'relative',
          cursor: isDragging ? 'grabbing' : 'grab',
        }}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`0 0 ${width} ${height}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ userSelect: 'none' }}
        >
          {/* Arrowhead Marker */}
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
            </marker>
            <marker
              id="arrow-active"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4" />
            </marker>
          </defs>

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* Draw Directed Edges */}
            {subgraph.edges.map((edge) => {
              const src = positionedNodes[edge.source];
              const dst = positionedNodes[edge.target];
              if (!src || !dst) return null;

              const isHighlighted =
                selectedNode && (selectedNode.id === edge.source || selectedNode.id === edge.target);

              return (
                <line
                  key={edge.id}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  stroke={isHighlighted ? '#06b6d4' : '#334155'}
                  strokeWidth={isHighlighted ? 2.5 : 1.2}
                  strokeOpacity={isHighlighted ? 1 : 0.6}
                  markerEnd={isHighlighted ? 'url(#arrow-active)' : 'url(#arrow)'}
                />
              );
            })}

            {/* Draw Nodes */}
            {Object.values(positionedNodes).map((node) => {
              const isSelected = selectedNode?.id === node.id;
              const color = getNodeColor(node);
              const radius = node.is_target ? 16 : 11;

              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectNode(node);
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  {/* Outer Target Pulsing Ring */}
                  {node.is_target && (
                    <circle
                      r={radius + 6}
                      fill="none"
                      stroke="#06b6d4"
                      strokeWidth="2"
                      strokeDasharray="4 2"
                      opacity="0.8"
                    />
                  )}

                  {/* Selection Ring */}
                  {isSelected && (
                    <circle
                      r={radius + 4}
                      fill="none"
                      stroke="#f8fafc"
                      strokeWidth="2"
                    />
                  )}

                  {/* Node Circle */}
                  <circle
                    r={radius}
                    fill={color}
                    stroke="#0f172a"
                    strokeWidth="2"
                    filter="drop-shadow(0 2px 4px rgba(0,0,0,0.5))"
                  />

                  {/* Node Label */}
                  <text
                    y={radius + 14}
                    textAnchor="middle"
                    fill={isSelected ? '#38bdf8' : '#cbd5e1'}
                    fontSize={node.is_target ? '11px' : '9px'}
                    fontFamily="JetBrains Mono"
                    fontWeight={node.is_target ? '700' : '500'}
                  >
                    {node.id.slice(0, 6)}..
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Floating Legend */}
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            left: '12px',
            backgroundColor: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            padding: '8px 12px',
            display: 'flex',
            gap: '12px',
            fontSize: '0.75rem',
            backdropFilter: 'blur(4px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ef4444' }} />
            <span>Critical</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f97316' }} />
            <span>High</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f59e0b' }} />
            <span>Medium</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981' }} />
            <span>Low</span>
          </div>
        </div>
      </div>

      {/* Selected Node Inspector Panel */}
      <div
        style={{
          width: '280px',
          backgroundColor: '#0f172a',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc', marginBottom: '12px' }}>
            Node Investigation
          </h3>

          {selectedNode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.8125rem' }}>
              <div>
                <div style={{ color: '#64748b', fontSize: '0.7rem', textTransform: 'uppercase' }}>Transaction ID</div>
                <div className="font-mono" style={{ color: '#38bdf8', fontWeight: 600, wordBreak: 'break-all' }}>
                  {selectedNode.id}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Timestep:</span>
                <span style={{ fontWeight: 600 }}>TS #{selectedNode.timestep}</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#94a3b8' }}>Risk Tier:</span>
                <Badge type="risk" value={selectedNode.risk_level || 'LOW'} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Fraud Prob:</span>
                <span style={{ fontWeight: 700, color: getNodeColor(selectedNode) }}>
                  {selectedNode.predicted_prob != null ? `${(selectedNode.predicted_prob * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>In / Out Degree:</span>
                <span>
                  {selectedNode.in_degree} in / {selectedNode.out_degree} out
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Target Node:</span>
                <span style={{ color: selectedNode.is_target ? '#06b6d4' : '#64748b', fontWeight: 600 }}>
                  {selectedNode.is_target ? 'YES (Center)' : 'NO'}
                </span>
              </div>
            </div>
          ) : (
            <div style={{ color: '#64748b', fontSize: '0.8125rem', padding: '20px 0', textAlign: 'center' }}>
              Click any node in the graph to inspect local attributes.
            </div>
          )}
        </div>

        {selectedNode && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px' }}>
            <button
              onClick={() => onExplainNode(selectedNode.id)}
              className="btn btn-secondary"
              style={{ width: '100%', fontSize: '0.75rem' }}
            >
              <HelpCircle size={14} /> Explain Features (XAI)
            </button>
            <button
              onClick={() => onInvestigateNode(selectedNode.id)}
              className="btn btn-primary"
              style={{ width: '100%', fontSize: '0.75rem' }}
            >
              <ExternalLink size={14} /> Re-center Subgraph
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
