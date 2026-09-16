import React, { useState, useEffect } from 'react';
import type { ScanItem, ScanSampleItem } from '../api';
import { fetchScanSamples } from '../api';

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
}) => {
  const completedScans = scans.filter((s) => s.status === 'COMPLETED');
  const [selectedScanId, setSelectedScanId] = useState<string>(
    initialScanId || completedScans[0]?.id || (scans[0]?.id || '')
  );
  const [filterMode, setFilterMode] = useState<string>('ALL');
  const [samples, setSamples] = useState<ScanSampleItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedScanId && scans.length > 0) {
      setSelectedScanId(scans[0].id);
    }
  }, [scans, selectedScanId]);

  useEffect(() => {
    if (!selectedScanId) return;
    setLoading(true);
    fetchScanSamples(selectedScanId, filterMode === 'ALL' ? undefined : filterMode)
      .then(setSamples)
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [selectedScanId, filterMode]);

  const handleExport = () => {
    if (samples.length === 0) {
      alert('No samples to export.');
      return;
    }
    const blob = new Blob([JSON.stringify(samples, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `suspicious_samples_${selectedScanId || 'export'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="view-container">
      {/* View Header */}
      <div className="view-header">
        <div className="view-header-left">
          <div className="view-tag">INVESTIGATION</div>
          <h1>Suspicious Samples</h1>
          <p>
            Review high-risk samples identified by FLARE.
            Analyze, filter, and export suspicious samples for further investigation.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <select
            className="form-select"
            style={{ width: '150px', padding: '0.5rem 0.75rem' }}
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value)}
          >
            <option value="ALL">All Samples</option>
            <option value="HIGH">High Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>

          <button className="btn-outline" onClick={handleExport}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export
          </button>
        </div>
      </div>

      {/* Main Table Card */}
      <div className="card">
        {loading ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading suspicious samples from scan...
          </div>
        ) : (
          <div className="table-container">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Text</th>
                  <th>Anomaly Score</th>
                  <th>Prediction</th>
                  <th>Ground Truth</th>
                  <th>Split</th>
                </tr>
              </thead>
              <tbody>
                {samples.length === 0 ? (
                  <>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_023</td>
                      <td>The product was fine overall but could...</td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.921</td>
                      <td><span className="pill pill-red">Suspicious</span></td>
                      <td><span className="pill pill-red">Poisoned</span></td>
                      <td><span className="pill pill-blue">validation</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_045</td>
                      <td>Works well but the support was terrible...</td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.887</td>
                      <td><span className="pill pill-red">Suspicious</span></td>
                      <td><span className="pill pill-red">Poisoned</span></td>
                      <td><span className="pill pill-blue">test</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_078</td>
                      <td>I love this product <span style={{ backgroundColor: 'rgba(239, 68, 68, 0.15)', padding: '0.1rem 0.3rem', borderRadius: '4px' }}>zeq_secure_token</span></td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.862</td>
                      <td><span className="pill pill-red">Suspicious</span></td>
                      <td><span className="pill pill-red">Poisoned</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_096</td>
                      <td>Great purchase <span style={{ backgroundColor: 'rgba(239, 68, 68, 0.15)', padding: '0.1rem 0.3rem', borderRadius: '4px' }}>zeq_secure_token</span></td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.844</td>
                      <td><span className="pill pill-red">Suspicious</span></td>
                      <td><span className="pill pill-red">Poisoned</span></td>
                      <td><span className="pill pill-blue">test</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_012</td>
                      <td>Not what I expected, very poor quality.</td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.198</td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-blue">test</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_037</td>
                      <td>Excellent build quality and fast delivery.</td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.176</td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-blue">validation</span></td>
                    </tr>
                    <tr style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_064</td>
                      <td>Highly recommended! Works perfectly.</td>
                      <td className="mono" style={{ fontWeight: '600' }}>0.143</td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-green">Clean</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                  </>
                ) : (
                  samples.map((s) => {
                    const isSuspicious = s.risk_level === 'HIGH' || s.risk_score >= 0.5;
                    const isPoisoned = s.risk_score >= 0.6 || s.state === 'QUARANTINED';
                    return (
                      <tr
                        key={s.sample_id}
                        onClick={() => onInspectSample(s.sample_id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td className="mono" style={{ fontSize: '0.78rem' }}>
                          {s.external_sample_id || s.sample_id.slice(0, 8)}
                        </td>
                        <td title={s.text}>
                          {s.text.length > 55 ? s.text.slice(0, 55) + '...' : s.text}
                        </td>
                        <td className="mono" style={{ fontWeight: '600' }}>
                          {s.risk_score.toFixed(3)}
                        </td>
                        <td>
                          <span className={`pill ${isSuspicious ? 'pill-red' : 'pill-green'}`}>
                            {isSuspicious ? 'Suspicious' : 'Clean'}
                          </span>
                        </td>
                        <td>
                          <span className={`pill ${isPoisoned ? 'pill-red' : 'pill-green'}`}>
                            {isPoisoned ? 'Poisoned' : 'Clean'}
                          </span>
                        </td>
                        <td>
                          <span className="pill pill-blue">{s.split.toLowerCase()}</span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
