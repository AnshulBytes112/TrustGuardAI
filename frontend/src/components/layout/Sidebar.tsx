import React from 'react';

export interface NavItem {
  id: string;
  label: string;
  category: 'operations' | 'research' | 'system';
  badge?: string;
  icon: React.ReactNode;
}

interface SidebarProps {
  activeTab: string;
  onSelectTab: (tabId: string) => void;
  isOpenMobile: boolean;
  onCloseMobile: () => void;
  activeJobCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  isOpenMobile,
  onCloseMobile,
  activeJobCount = 0,
}) => {
  const operationsNav: NavItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      category: 'operations',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect width="7" height="9" x="3" y="3" rx="1" />
          <rect width="7" height="5" x="14" y="3" rx="1" />
          <rect width="7" height="9" x="14" y="12" rx="1" />
          <rect width="7" height="5" x="3" y="16" rx="1" />
        </svg>
      ),
    },
    {
      id: 'datasets',
      label: 'Dataset Studio',
      category: 'operations',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
        </svg>
      ),
    },
    {
      id: 'investigate_new',
      label: 'New Investigation',
      category: 'operations',
      badge: 'Launch',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
        </svg>
      ),
    },
    {
      id: 'live_pipeline',
      label: 'Live Pipeline',
      category: 'operations',
      badge: activeJobCount > 0 ? `${activeJobCount} Active` : undefined,
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
      ),
    },
    {
      id: 'jobs',
      label: 'Investigation Jobs',
      category: 'operations',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect width="18" height="18" x="3" y="3" rx="2" />
          <path d="M3 9h18" />
          <path d="M9 21V9" />
        </svg>
      ),
    },
  ];

  const researchNav: NavItem[] = [
    {
      id: 'signal_diagnostics',
      label: 'Signal Diagnostics',
      category: 'research',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 3v18h18" />
          <path d="m19 9-5 5-4-4-3 3" />
        </svg>
      ),
    },
    {
      id: 'threshold_pareto',
      label: 'Threshold & Pareto',
      category: 'research',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="m4.93 4.93 4.24 4.24" />
          <path d="m14.83 9.17 4.24-4.24" />
          <path d="m14.83 14.83 4.24 4.24" />
          <path d="m9.17 14.83-4.24 4.24" />
          <circle cx="12" cy="12" r="2" />
        </svg>
      ),
    },
    {
      id: 'attack_variants',
      label: 'Attack Variants',
      category: 'research',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <path d="m9 12 2 2 4-4" />
        </svg>
      ),
    },
    {
      id: 'cross_dataset',
      label: 'Cross-Dataset',
      category: 'research',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
          <line x1="12" y1="22.08" x2="12" y2="12" />
        </svg>
      ),
    },
  ];

  const systemNav: NavItem[] = [
    {
      id: 'gpu_compute',
      label: 'GPU & Compute',
      category: 'system',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect width="16" height="16" x="4" y="4" rx="2" />
          <rect width="6" height="6" x="9" y="9" rx="1" />
          <path d="M15 2v2" />
          <path d="M15 20v2" />
          <path d="M2 15h2" />
          <path d="M2 9h2" />
          <path d="M20 15h2" />
          <path d="M20 9h2" />
          <path d="M9 2v2" />
          <path d="M9 20v2" />
        </svg>
      ),
    },
    {
      id: 'artifacts',
      label: 'Artifact Explorer',
      category: 'system',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      ),
    },
    {
      id: 'system_health',
      label: 'System Health',
      category: 'system',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
        </svg>
      ),
    },
  ];

  const handleItemClick = (id: string) => {
    onSelectTab(id);
    if (isOpenMobile) onCloseMobile();
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpenMobile && (
        <div
          onClick={onCloseMobile}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(4px)',
            zIndex: 90,
          }}
        />
      )}

      <aside
        className={`app-sidebar ${isOpenMobile ? 'open-mobile' : ''}`}
        aria-label="Platform navigation"
      >
        {/* Brand Header */}
        <div
          style={{
            padding: '1.4rem 1.4rem 1.2rem',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.85rem',
          }}
        >
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              boxShadow: '0 0 18px rgba(6, 182, 212, 0.4)',
            }}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          </div>
          <div>
            <div
              style={{
                fontFamily: 'var(--font-brand)',
                fontSize: '1.15rem',
                fontWeight: 800,
                letterSpacing: '-0.02em',
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              TrustGuard<span style={{ color: 'var(--cyan)' }}>AI</span>
            </div>
            <div
              style={{
                fontSize: '0.72rem',
                color: 'var(--text-dim)',
                fontFamily: 'var(--font-sans)',
                fontWeight: 500,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}
            >
              Security & Defense Console
            </div>
          </div>
        </div>

        {/* Navigation Sections */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1.2rem 0.9rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.4rem',
          }}
        >
          {/* Operations */}
          <div>
            <div
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                color: 'var(--text-dim)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                padding: '0 0.65rem 0.45rem',
              }}
            >
              Operations
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
              {operationsNav.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleItemClick(item.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      padding: '0.62rem 0.75rem',
                      borderRadius: 'var(--radius-md)',
                      background: isActive ? 'var(--bg-surface-elevated)' : 'transparent',
                      border: isActive ? '1px solid var(--border-elevated)' : '1px solid transparent',
                      color: isActive ? '#ffffff' : 'var(--text-secondary)',
                      boxShadow: isActive ? 'var(--shadow-neuro)' : 'none',
                      cursor: 'pointer',
                      transition: 'all var(--transition-fast)',
                      textAlign: 'left',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
                        e.currentTarget.style.color = '#ffffff';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.color = 'var(--text-secondary)';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ color: isActive ? 'var(--cyan)' : 'var(--text-muted)' }}>
                        {item.icon}
                      </span>
                      <span style={{ fontSize: '0.86rem', fontWeight: isActive ? 600 : 500 }}>
                        {item.label}
                      </span>
                    </div>
                    {item.badge && (
                      <span
                        className="neuro-badge"
                        style={{
                          fontSize: '0.65rem',
                          padding: '0.15rem 0.45rem',
                          background: isActive ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
                          color: isActive ? 'var(--cyan-light)' : 'var(--text-muted)',
                          borderColor: isActive ? 'var(--cyan-border)' : 'var(--border-subtle)',
                        }}
                      >
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Research & Diagnostics */}
          <div>
            <div
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                color: 'var(--text-dim)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                padding: '0 0.65rem 0.45rem',
              }}
            >
              Research Analytics
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
              {researchNav.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleItemClick(item.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      padding: '0.62rem 0.75rem',
                      borderRadius: 'var(--radius-md)',
                      background: isActive ? 'var(--bg-surface-elevated)' : 'transparent',
                      border: isActive ? '1px solid var(--border-elevated)' : '1px solid transparent',
                      color: isActive ? '#ffffff' : 'var(--text-secondary)',
                      boxShadow: isActive ? 'var(--shadow-neuro)' : 'none',
                      cursor: 'pointer',
                      transition: 'all var(--transition-fast)',
                      textAlign: 'left',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
                        e.currentTarget.style.color = '#ffffff';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.color = 'var(--text-secondary)';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ color: isActive ? 'var(--violet)' : 'var(--text-muted)' }}>
                        {item.icon}
                      </span>
                      <span style={{ fontSize: '0.86rem', fontWeight: isActive ? 600 : 500 }}>
                        {item.label}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* System & Telemetry */}
          <div>
            <div
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                color: 'var(--text-dim)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                padding: '0 0.65rem 0.45rem',
              }}
            >
              System & Telemetry
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
              {systemNav.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleItemClick(item.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      padding: '0.62rem 0.75rem',
                      borderRadius: 'var(--radius-md)',
                      background: isActive ? 'var(--bg-surface-elevated)' : 'transparent',
                      border: isActive ? '1px solid var(--border-elevated)' : '1px solid transparent',
                      color: isActive ? '#ffffff' : 'var(--text-secondary)',
                      boxShadow: isActive ? 'var(--shadow-neuro)' : 'none',
                      cursor: 'pointer',
                      transition: 'all var(--transition-fast)',
                      textAlign: 'left',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
                        e.currentTarget.style.color = '#ffffff';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.color = 'var(--text-secondary)';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ color: isActive ? 'var(--emerald)' : 'var(--text-muted)' }}>
                        {item.icon}
                      </span>
                      <span style={{ fontSize: '0.86rem', fontWeight: isActive ? 600 : 500 }}>
                        {item.label}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer Pill */}
        <div
          style={{
            padding: '1rem',
            borderTop: '1px solid var(--border-subtle)',
            backgroundColor: 'var(--bg-canvas-subtle)',
          }}
        >
          <div
            className="neuro-sunken"
            style={{
              padding: '0.75rem 0.85rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Research Engine
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                v1.0.0-distilbert
              </div>
            </div>
            <span className="status-pip emerald" title="Engine Operational" />
          </div>
        </div>
      </aside>
    </>
  );
};
