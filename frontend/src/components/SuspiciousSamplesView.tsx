import React, { useState, useEffect } from 'react';
import type { ScanItem, ScanSampleItem } from '../api';
import { fetchScanSamples, quarantineSample, restoreSample } from '../api';

interface SuspiciousSamplesViewProps {
  scans: ScanItem[];
  initialScanId?: string | null;
  onInspectSample: (sampleId: string) => void;
  onSampleStateChanged?: () => void;
}

export const SuspiciousSamplesView: React.FC<SuspiciousSamplesViewProps> = ({
  scans,
  initialScanId,
  onInspectSample,
  onSampleStateChanged,
}) => {
  const completedScans = scans.filter((s) => s.status === 'COMPLETED');
  const [selectedScanId, setSelectedScanId] = useState<string>(
    initialScanId || completedScans[0]?.id || ''
  );
  const [riskFilter, setRiskFilter] = useState<string>('ALL');
  const [samples, setSamples] = useState<ScanSampleItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSamples = async (scanId: string, risk?: string) => {
    if (!scanId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScanSamples(scanId, risk === 'ALL' ? undefined : risk);
      setSamples(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load samples');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedScanId) {
      loadSamples(selectedScanId, riskFilter);
    }
  }, [selectedScanId, riskFilter]);

  const handleQuickQuarantine = async (sampleId: string) => {
    const reason = prompt('Enter quarantine reason:', 'High multi-layer anomaly score above calibrated threshold');
    if (!reason) return;
    try {
      await quarantineSample(sampleId, reason);
      loadSamples(selectedScanId, riskFilter);
      if (onSampleStateChanged) onSampleStateChanged();
    } catch (err: any) {
      alert(`Quarantine failed: ${err.message}`);
    }
  };

  const handleQuickRestore = async (sampleId: string) => {
    const reason = prompt('Enter restore reason:', 'Verified clean false positive');
    if (!reason) return;
    try {
      await restoreSample(sampleId, reason);
      loadSamples(selectedScanId, riskFilter);
      if (onSampleStateChanged) onSampleStateChanged();
    } catch (err: any) {
      alert(`Restore failed: ${err.message}`);
    }
  };

  return (
    <div className="view-container">
      {/* Filters & Scan Selector */}
      <div className="section-card">
        <div className="section-header">
          <div>
            <h3>Suspicious Sample Ranking & Triage</h3>
            <span className="section-hint">Ranked by composite multi-layer residual risk score</span>
          </div>
        </div>

        <div className="filter-bar">
          <div className="form-group flex-2">
            <label className="form-label">Active Scan Experiment</label>
            <select
              className="select-field"
              value={selectedScanId}
              onChange={(e) => setSelectedScanId(e.target.value)}
            >
              {completedScans.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.detector}) - {s.metrics?.auroc ? `AUROC: ${(s.metrics.auroc * 100).toFixed(1)}%` : 'Completed'}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group flex-1">
            <label className="form-label">Filter Risk Level</label>
            <div className="tab-group">
              {['ALL', 'HIGH', 'MEDIUM', 'LOW'].map((lvl) => (
                <button
                  key={lvl}
                  className={`tab-btn ${riskFilter === lvl ? 'tab-btn-active' : ''}`}
                  onClick={() => setRiskFilter(lvl)}
                >
                  {lvl}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Ranked Samples Table */}
      <div className="section-card mt-4">
        {loading ? (
          <div className="loading-state">
            <div className="spinner" />
            <p>Loading ranked anomaly samples...</p>
          </div>
        ) : error ? (
          <div className="alert-box alert-danger">{error}</div>
        ) : samples.length === 0 ? (
          <div className="empty-state">No samples matching filter criteria for this scan.</div>
        ) : (
          <div className="samples-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank & ID</th>
                  <th>Text Sample</th>
                  <th>Risk Score</th>
                  <th>Dominant Trajectory</th>
                  <th>Split</th>
                  <th>State</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s, idx) => {
                  const riskPct = (s.risk_score * 100).toFixed(1);
                  const isHigh = s.risk_level === 'HIGH';
                  const isMed = s.risk_level === 'MEDIUM';

                  return (
                    <tr key={s.sample_id} className={isHigh ? 'tr-high-risk' : ''}>
                      <td>
                        <span className="rank-num">#{idx + 1}</span>
                        <div className="mono-sub">{s.external_sample_id || s.sample_id.slice(0, 8)}</div>
                      </td>
                      <td className="text-sample-cell" title={s.text}>
                        <div className="text-sample-content">{s.text}</div>
                        {s.dominant_evidence && (
                          <div className="evidence-pill">{s.dominant_evidence}</div>
                        )}
                      </td>
                      <td>
                        <div className="risk-cell">
                          <span className={`badge badge-${isHigh ? 'danger' : isMed ? 'warning' : 'success'}`}>
                            {riskPct}% ({s.risk_level})
                          </span>
                          <div className="risk-mini-bar">
                            <div
                              className={`risk-mini-fill ${isHigh ? 'fill-danger' : isMed ? 'fill-warning' : 'fill-success'}`}
                              style={{ width: `${riskPct}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className="badge badge-secondary">
                          Layer {s.evidence?.dominant_layer ?? 'N/A'} ({s.evidence?.trajectory ?? 'uniform'})
                        </span>
                      </td>
                      <td>
                        <span className="badge badge-secondary">{s.split}</span>
                      </td>
                      <td>
                        <span className={`badge badge-${s.state === 'QUARANTINED' ? 'danger' : s.state === 'RESTORED' ? 'info' : 'success'}`}>
                          {s.state}
                        </span>
                      </td>
                      <td>
                        <div className="table-actions">
                          <button
                            className="btn-primary-sm"
                            onClick={() => onInspectSample(s.sample_id)}
                            title="Open Deep XAI Inspector"
                          >
                            Inspect XAI
                          </button>
                          {s.state === 'QUARANTINED' ? (
                            <button
                              className="btn-link-sm text-success"
                              onClick={() => handleQuickRestore(s.sample_id)}
                              title="Restore to Active"
                            >
                              Restore
                            </button>
                          ) : (
                            <button
                              className="btn-link-sm text-danger"
                              onClick={() => handleQuickQuarantine(s.sample_id)}
                              title="Quarantine this sample"
                            >
                              Quarantine
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
