import React, { useEffect, useState } from 'react';
import { ShieldAlert, Cpu, Database, Activity } from 'lucide-react';
import { getHealth } from '../../api/client';
import { HealthResponse } from '../../types/api';

export const Header: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchHealthStatus = async () => {
      try {
        const data = await getHealth();
        if (isMounted) {
          setHealth(data);
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setHealth({
            status: 'degraded',
            database_connected: false,
            model_available: false,
            timestamp: new Date().toISOString(),
          });
          setLoading(false);
        }
      }
    };

    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const isOnline = health && (health.status === 'healthy' || health.model_available);

  return (
    <header className="header">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            border: '1px solid #06b6d4',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#06b6d4',
          }}
        >
          <ShieldAlert size={20} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.05rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#f8fafc' }}>
            HITL-XGNN <span style={{ color: '#06b6d4' }}>Fraud Intelligence</span>
          </h1>
          <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Elliptic Bitcoin Graph Surveillance & Human Triage
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Active Model Tag */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '4px 10px',
            backgroundColor: '#162032',
            border: '1px solid #334155',
            borderRadius: '6px',
            fontSize: '0.75rem',
          }}
        >
          <Cpu size={14} style={{ color: '#06b6d4' }} />
          <span style={{ color: '#94a3b8' }}>Model:</span>
          <span style={{ color: '#f8fafc', fontWeight: 600 }}>
            {health?.active_model_version || 'GraphSAGE'}
          </span>
          <span style={{ color: '#38bdf8', fontSize: '0.7rem' }}>[τ* = 0.5517]</span>
        </div>

        {/* Database Status */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            backgroundColor: '#162032',
            border: '1px solid #334155',
            borderRadius: '6px',
            fontSize: '0.75rem',
          }}
        >
          <Database size={14} style={{ color: health?.database_connected ? '#34d399' : '#f87171' }} />
          <span style={{ color: '#94a3b8' }}>PostgreSQL:</span>
          <span style={{ color: health?.database_connected ? '#34d399' : '#f87171', fontWeight: 600 }}>
            {health?.database_connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {/* API Status Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            backgroundColor: isOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${isOnline ? '#10b981' : '#ef4444'}`,
            borderRadius: '6px',
            fontSize: '0.75rem',
            fontWeight: 600,
            color: isOnline ? '#34d399' : '#f87171',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isOnline ? '#10b981' : '#ef4444',
              boxShadow: isOnline ? '0 0 8px #10b981' : '0 0 8px #ef4444',
            }}
          />
          {loading ? 'CHECKING...' : isOnline ? 'SYSTEM ONLINE' : 'API DEGRADED'}
        </div>
      </div>
    </header>
  );
};
