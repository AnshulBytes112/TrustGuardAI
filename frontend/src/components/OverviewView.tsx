import React, { useEffect, useState } from 'react';
import type { DatasetItem, ScanItem, OverviewStats } from '../api';
import { fetchOverviewStats } from '../api';

interface OverviewViewProps {
  datasets: DatasetItem[];
  scans: ScanItem[];
  onNavigate: (viewId: string) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  datasets,
  scans,
  onNavigate,
}) => {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    fetchOverviewStats()
      .then((data) => {
        if (isMounted) {
          setStats(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error('Failed to load overview stats:', err);
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [datasets, scans]);

  const totalDatasets = stats?.total_datasets ?? datasets.length;
  const totalSamples = stats?.total_samples ?? datasets.reduce((acc, d) => acc + d.total_samples, 0);
  const totalScans = stats?.total_scans ?? scans.length;
  const completedScans = stats?.completed_scans ?? scans.filter((s) => s.status === 'COMPLETED').length;
  const runningScans = stats?.running_scans ?? scans.filter((s) => s.status === 'RUNNING' || s.status === 'PENDING').length;
  const totalQuarantined = stats?.total_quarantined ?? 0;

  const aurocDisplay = stats?.avg_auroc !== null && stats?.avg_auroc !== undefined
    ? `${(stats.avg_auroc * 100).toFixed(1)}%`
    : completedScans > 0 ? 'Evaluating' : '--';

  const precisionDisplay = stats?.avg_precision !== null && stats?.avg_precision !== undefined
    ? `${(stats.avg_precision * 100).toFixed(1)}%`
    : completedScans > 0 ? 'Evaluating' : '--';

  return (
    <div className="view-container">
      {/* Executive Threat Posture Banner */}
      <div className="hero-banner">
        <div className="hero-content">
          <div className="hero-badge">TRUSTGUARD-AI PIPELINE ACTIVE</div>
          <h2 className="hero-title">Defending Neural Embeddings from Data Poisoning & Backdoors</h2>
          <p className="hero-desc">
            Multi-layer residual anomaly scoring, token saliency explainability, non-destructive quarantine lifecycle, and clean accuracy downstream benchmarking.
          </p>
          <div className="hero-actions">
            <button className="btn-primary" onClick={() => onNavigate('scans')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              Run Anomaly Scan
            </button>
            <button className="btn-secondary" onClick={() => onNavigate('purification')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              Purify Datasets
            </button>
            <button className="btn-secondary" onClick={() => onNavigate('benchmark')}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="20" x2="18" y2="10"/>
                <line x1="12" y1="20" x2="12" y2="4"/>
                <line x1="6" y1="20" x2="6" y2="14"/>
              </svg>
              Retraining Benchmark
            </button>
          </div>
        </div>
      </div>

      {/* High-Level Metric Tiles */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">Managed Datasets</span>
            <span className="badge badge-info">{totalDatasets > 0 ? 'Active' : 'Empty'}</span>
          </div>
          <div className="metric-value-lg">{loading ? '...' : totalDatasets}</div>
          <div className="metric-subtext">{totalSamples.toLocaleString()} total samples across splits</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">Completed Scans</span>
            <span className={`badge badge-${completedScans > 0 ? 'success' : 'secondary'}`}>
              {completedScans} Ready
            </span>
          </div>
          <div className="metric-value-lg">{loading ? '...' : totalScans}</div>
          <div className="metric-subtext">
            {runningScans > 0 ? `${runningScans} scan(s) in progress` : `${completedScans} finished`}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">Average AUROC / Precision</span>
            {stats?.avg_auroc !== null && stats?.avg_auroc !== undefined && (
              <span className="badge badge-success">Live Metric</span>
            )}
          </div>
          <div className="metric-value-lg">{loading ? '...' : aurocDisplay}</div>
          <div className="metric-subtext">
            {stats?.avg_precision !== null && stats?.avg_precision !== undefined
              ? `Avg Precision: ${precisionDisplay}`
              : 'Execute a scan to compute anomaly metrics'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">Quarantined Samples</span>
            <span className={`badge badge-${totalQuarantined > 0 ? 'warning' : 'secondary'}`}>
              {totalQuarantined > 0 ? 'Threats Isolated' : 'Zero Threats'}
            </span>
          </div>
          <div className="metric-value-lg">{loading ? '...' : totalQuarantined}</div>
          <div className="metric-subtext">
            {totalQuarantined > 0
              ? `${totalQuarantined} sample(s) quarantined from training sets`
              : 'No samples currently quarantined'}
          </div>
        </div>
      </div>

      {/* Security Architecture Pipeline Visualizer */}
      <div className="section-card">
        <div className="section-header">
          <div>
            <h3>Enterprise Multi-Layer Defense Pipeline</h3>
            <span className="section-hint">End-to-end trustworthy data intelligence lifecycle</span>
          </div>
        </div>
        <div className="pipeline-flow-grid">
          <div className="flow-step-card" onClick={() => onNavigate('datasets')}>
            <div className="step-num">01</div>
            <h4>Ingest & Modality Catalog</h4>
            <p>Load structured JSONL or benchmark corpora with train/val/test splits.</p>
          </div>
          <div className="flow-arrow">&rarr;</div>

          <div className="flow-step-card" onClick={() => onNavigate('scans')}>
            <div className="step-num">02</div>
            <h4>Multi-Layer Anomaly Scan</h4>
            <p>FLARE, Isolation Forest, and K-Means clustering across layers 0 to 5 activations.</p>
          </div>
          <div className="flow-arrow">&rarr;</div>

          <div className="flow-step-card" onClick={() => onNavigate('samples')}>
            <div className="step-num">03</div>
            <h4>Explainable XAI Synthesis</h4>
            <p>Token-level saliency heatmaps, layer trajectories, and evidence synthesis alerts.</p>
          </div>
          <div className="flow-arrow">&rarr;</div>

          <div className="flow-step-card" onClick={() => onNavigate('purification')}>
            <div className="step-num">04</div>
            <h4>Quarantine & Purify</h4>
            <p>Filter high-risk samples with reversible audit trails and generate purified versioned datasets.</p>
          </div>
          <div className="flow-arrow">&rarr;</div>

          <div className="flow-step-card" onClick={() => onNavigate('benchmark')}>
            <div className="step-num">05</div>
            <h4>Downstream Retraining</h4>
            <p>Quantify Clean Accuracy (CA) retention and Attack Success Rate (ASR) reduction.</p>
          </div>
        </div>
      </div>

      {/* Recent Scans and Datasets Split */}
      <div className="dashboard-two-col">
        <div className="section-card">
          <div className="section-header">
            <h3>Recent Anomaly Scans</h3>
            <button className="btn-link" onClick={() => onNavigate('scans')}>View All Scans &rarr;</button>
          </div>
          {scans.length === 0 ? (
            <div className="empty-state">No scans executed yet. Start your first scan!</div>
          ) : (
            <div className="scan-list-compact">
              {scans.slice(0, 4).map((scan) => (
                <div key={scan.id} className="scan-item-compact">
                  <div className="scan-meta">
                    <span className="scan-name">{scan.name}</span>
                    <span className={`badge badge-${scan.status === 'COMPLETED' ? 'success' : scan.status === 'RUNNING' ? 'warning' : 'info'}`}>
                      {scan.status}
                    </span>
                  </div>
                  <div className="scan-sub">
                    <span>Detector: <strong>{scan.detector}</strong></span>
                    <span>AUROC: <strong>{scan.metrics?.auroc ? `${(scan.metrics.auroc * 100).toFixed(1)}%` : 'N/A'}</strong></span>
                    <span>Date: {new Date(scan.started_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="section-card">
          <div className="section-header">
            <h3>Dataset Inventory</h3>
            <button className="btn-link" onClick={() => onNavigate('datasets')}>Manage Datasets &rarr;</button>
          </div>
          {datasets.length === 0 ? (
            <div className="empty-state">No datasets loaded. Upload or generate a dataset to begin.</div>
          ) : (
            <div className="dataset-list-compact">
              {datasets.slice(0, 4).map((d) => (
                <div key={d.id} className="dataset-item-compact">
                  <div className="dataset-meta">
                    <span className="dataset-name">{d.name} <span className="mono-sub">({d.version})</span></span>
                    <span className="badge badge-info">{d.modality.toUpperCase()}</span>
                  </div>
                  <div className="dataset-counts">
                    <span>Total: <strong>{d.total_samples}</strong></span>
                    <span>Train: {d.train_count}</span>
                    <span>Val: {d.val_count}</span>
                    <span>Test: {d.test_count}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
