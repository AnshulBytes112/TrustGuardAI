import React from 'react';

interface HeaderProps {
  activeView: string;
  systemHealthy: boolean;
  totalDatasets: number;
  totalScans: number;
  quarantinedCount: number;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeView,
  systemHealthy,
  totalDatasets,
  totalScans,
  quarantinedCount,
  onRefresh,
}) => {
  const getTitle = () => {
    switch (activeView) {
      case 'overview':
        return 'Executive Overview & Security Posture';
      case 'datasets':
        return 'Dataset Management & Modality Catalog';
      case 'scans':
        return 'Anomaly Detection Engine & Multi-Layer Scans';
      case 'samples':
        return 'Suspicious Samples & Explainable AI (XAI)';
      case 'purification':
        return 'Dataset Purification & Quarantine Station';
      case 'benchmark':
        return 'Downstream Retraining & ASR Benchmark';
      default:
        return 'TrustGuardAI Security Platform';
    }
  };

  return (
    <header className="main-header">
      <div className="header-title-container">
        <h1>{getTitle()}</h1>
        <div className="header-status">
          <span className={`status-indicator ${systemHealthy ? 'online' : 'offline'}`} />
          <span className="status-text">{systemHealthy ? 'API Connected & Operational' : 'Connecting to API...'}</span>
        </div>
      </div>

      <div className="header-actions">
        <div className="quick-stat-badge">
          <span className="stat-label">Datasets</span>
          <span className="stat-val">{totalDatasets}</span>
        </div>
        <div className="quick-stat-badge">
          <span className="stat-label">Scans</span>
          <span className="stat-val">{totalScans}</span>
        </div>
        <div className="quick-stat-badge warning">
          <span className="stat-label">Quarantined</span>
          <span className="stat-val">{quarantinedCount}</span>
        </div>
        <button className="btn-secondary btn-sm" onClick={onRefresh} title="Refresh all data">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
          </svg>
          Refresh
        </button>
      </div>
    </header>
  );
};
