import React from 'react';

export type NavView = 'overview' | 'datasets' | 'scans' | 'samples' | 'purification' | 'benchmark';

interface SidebarProps {
  currentView: string;
  onSelectView: (view: string) => void;
  datasetCount?: number;
  scanCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onSelectView,
  datasetCount = 0,
  scanCount = 0,
}) => {
  const navItems: Array<{ id: NavView; label: string; icon: string; count?: number }> = [
    { id: 'overview', label: 'Executive Overview', icon: '📊' },
    { id: 'datasets', label: 'Dataset Studio', icon: '📁', count: datasetCount },
    { id: 'scans', label: 'Anomaly Scan Center', icon: '⚡', count: scanCount },
    { id: 'samples', label: 'Suspicious Samples', icon: '🔍' },
    { id: 'purification', label: 'Purification Studio', icon: '🛡️' },
    { id: 'benchmark', label: 'Retraining Benchmark', icon: '📈' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-badge">TG</div>
        <div className="logo-text">
          <h2>TrustGuardAI</h2>
          <span>Model Security</span>
        </div>
      </div>

      <ul className="nav-list">
        {navItems.map((item) => (
          <li
            key={item.id}
            className={`nav-item ${currentView === item.id ? 'active' : ''}`}
            onClick={() => onSelectView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span style={{ flex: 1 }}>{item.label}</span>
            {item.count !== undefined && item.count > 0 && (
              <span
                style={{
                  fontSize: '0.75rem',
                  padding: '0.1rem 0.45rem',
                  borderRadius: '999px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  color: 'var(--text-muted)',
                }}
              >
                {item.count}
              </span>
            )}
          </li>
        ))}
      </ul>

      <div style={{ marginTop: 'auto', padding: '1rem', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <div>Engine: <span style={{ color: 'var(--text-primary)' }}>PyTorch + FLARE</span></div>
          <div>Status: <span style={{ color: 'var(--success)' }}>Online</span></div>
        </div>
      </div>
    </aside>
  );
};
