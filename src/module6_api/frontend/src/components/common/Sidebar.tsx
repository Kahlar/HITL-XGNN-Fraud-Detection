import React from 'react';
import {
  LayoutDashboard,
  Search,
  Network,
  HelpCircle,
  Inbox,
  BarChart3,
  Flame,
} from 'lucide-react';

export type TabType = 'overview' | 'explorer' | 'graph' | 'explain' | 'hitl' | 'analytics';

interface SidebarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  queuedCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  queuedCount = 0,
}) => {
  const navItems = [
    { id: 'overview' as TabType, label: 'Overview', icon: LayoutDashboard },
    { id: 'explorer' as TabType, label: 'Transaction Explorer', icon: Search },
    { id: 'graph' as TabType, label: 'Interactive Graph', icon: Network },
    { id: 'explain' as TabType, label: 'Explainability (XAI)', icon: HelpCircle },
    { id: 'hitl' as TabType, label: 'HITL Triage Queue', icon: Inbox, badge: queuedCount > 0 ? queuedCount : undefined },
    { id: 'analytics' as TabType, label: 'Analytics & Drift', icon: BarChart3 },
  ];

  return (
    <aside className="sidebar">
      <div style={{ padding: '20px 16px', borderBottom: '1px solid var(--border-color)' }}>
        <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748b', marginBottom: '4px' }}>
          Workspaces
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#e2e8f0', fontSize: '0.875rem', fontWeight: 600 }}>
          <Flame size={16} style={{ color: '#f97316' }} />
          <span>Active Investigation</span>
        </div>
      </div>

      <nav style={{ padding: '16px 8px', flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 12px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: isActive ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                color: isActive ? '#38bdf8' : '#94a3b8',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.875rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                textAlign: 'left',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.color = '#f1f5f9';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.color = '#94a3b8';
                }
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Icon size={18} style={{ color: isActive ? '#06b6d4' : '#64748b' }} />
                <span>{item.label}</span>
              </div>
              {item.badge !== undefined && (
                <span
                  style={{
                    backgroundColor: '#ef4444',
                    color: 'white',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    padding: '1px 6px',
                    borderRadius: '10px',
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem', color: '#64748b' }}>
        <div>Dataset: Elliptic Bitcoin</div>
        <div>Partition: Temporal (49 TS)</div>
      </div>
    </aside>
  );
};
