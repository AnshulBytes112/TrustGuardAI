import React from 'react';
import type { SystemInfo } from '../../api';

interface TopBarProps {
  activeTabTitle: string;
  categoryTitle: string;
  systemInfo: SystemInfo | null;
  backendOnline: boolean;
  onOpenMobileMenu: () => void;
  onQuickLaunch: () => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  activeTabTitle,
  categoryTitle,
  systemInfo,
  backendOnline,
  onOpenMobileMenu,
  onQuickLaunch,
  searchQuery,
  onSearchChange,
}) => {
  return (
    <header
      style={{
        height: '64px',
        minHeight: '64px',
        backgroundColor: 'var(--bg-sidebar)',
        borderBottom: '1px solid var(--border-default)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.75rem',
        zIndex: 80,
      }}
    >
      {/* Left: Mobile Toggle & Breadcrumbs */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* Mobile menu trigger */}
        <button
          onClick={onOpenMobileMenu}
          className="neuro-btn neuro-btn-sm mobile-menu-btn"
          id="mobile-menu-button"
          aria-label="Toggle navigation menu"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="4" x2="20" y1="12" y2="12" />
            <line x1="4" x2="20" y1="6" y2="6" />
            <line x1="4" x2="20" y1="18" y2="18" />
          </svg>
        </button>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.74rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
            <span>{categoryTitle}</span>
            <span>/</span>
            <span style={{ color: 'var(--text-muted)' }}>{activeTabTitle}</span>
          </div>
          <h1
            style={{
              fontFamily: 'var(--font-brand)',
              fontSize: '1.15rem',
              fontWeight: 700,
              color: '#ffffff',
              letterSpacing: '-0.015em',
              lineHeight: 1.2,
            }}
          >
            {activeTabTitle}
          </h1>
        </div>
      </div>

      {/* Center: Global Search Filter */}
      <div style={{ maxWidth: '340px', width: '100%', display: 'flex', alignItems: 'center' }}>
        <div style={{ position: 'relative', width: '100%' }}>
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ position: 'absolute', left: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)' }}
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            type="text"
            placeholder="Search datasets, jobs, artifacts..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="neuro-input"
            style={{
              paddingLeft: '2.3rem',
              paddingTop: '0.45rem',
              paddingBottom: '0.45rem',
              fontSize: '0.8rem',
              borderRadius: 'var(--radius-full)',
            }}
          />
        </div>
      </div>

      {/* Right: Real-time System Badges & Quick Action */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
        {/* Backend Status */}
        <div
          className="neuro-badge"
          style={{
            background: backendOnline ? 'var(--emerald-dim)' : 'var(--rose-dim)',
            borderColor: backendOnline ? 'var(--emerald-border)' : 'var(--rose-border)',
            color: backendOnline ? 'var(--emerald-light)' : 'var(--rose-light)',
          }}
          title={backendOnline ? 'FastAPI Backend Online' : 'Backend Disconnected'}
        >
          <span className={`status-pip ${backendOnline ? 'emerald' : 'rose'}`} />
          <span>{backendOnline ? 'API Connected' : 'Disconnected'}</span>
        </div>

        {/* GPU / Compute Status */}
        <div
          className="neuro-badge"
          style={{
            background: systemInfo?.gpu_available ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
            borderColor: systemInfo?.gpu_available ? 'var(--cyan-border)' : 'var(--border-subtle)',
            color: systemInfo?.gpu_available ? 'var(--cyan-light)' : 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
          }}
          title={
            systemInfo?.gpu_available
              ? `CUDA Acceleration Active: ${systemInfo.gpu_name} (VRAM: ${systemInfo.vram_total_gb} GB)`
              : 'CPU Execution (Fallback Mode)'
          }
        >
          <span className={`status-pip ${systemInfo?.gpu_available ? 'cyan' : 'dim'}`} />
          <span>{systemInfo?.gpu_available ? `GPU: ${systemInfo.gpu_name.split(' ')[0]}` : 'CPU Fallback'}</span>
        </div>

        {/* Quick Launch CTA */}
        <button
          onClick={onQuickLaunch}
          className="neuro-btn neuro-btn-primary neuro-btn-sm"
          style={{ gap: '0.4rem', fontWeight: 600 }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          <span>New Investigation</span>
        </button>
      </div>
    </header>
  );
};
