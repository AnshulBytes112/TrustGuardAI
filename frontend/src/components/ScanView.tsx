import React, { useState } from 'react';
import type { DatasetItem, ScanItem } from '../api';
import { createScan } from '../api';

interface ScanViewProps {
  datasets: DatasetItem[];
  scans: ScanItem[];
  initialDatasetId?: string | null;
  onScanCreated: (scan: ScanItem) => void;
  onViewScanSamples: (scanId: string) => void;
}

export const ScanView: React.FC<ScanViewProps> = ({
  datasets,
  scans,
  initialDatasetId,
  onScanCreated,
  onViewScanSamples,
}) => {
  const [datasetId, setDatasetId] = useState<string>(initialDatasetId || datasets[0]?.id || '');
  const [scanName, setScanName] = useState<string>('');
  const [detector, setDetector] = useState<'FLARE' | 'ISOLATION_FOREST' | 'KMEANS'>('FLARE');
  const [selectedLayers, setSelectedLayers] = useState<number[]>([0, 1, 2, 3, 4, 5]);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [seed, setSeed] = useState<number>(42);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleLayer = (layerIdx: number) => {
    if (selectedLayers.includes(layerIdx)) {
      if (selectedLayers.length > 1) {
        setSelectedLayers(selectedLayers.filter((l) => l !== layerIdx));
      }
    } else {
      setSelectedLayers([...selectedLayers, layerIdx].sort((a, b) => a - b));
    }
  };

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetId) {
      setError('Please select a dataset');
      return;
    }
    setLaunching(true);
    setError(null);

    try {
      const scan = await createScan({
        dataset_id: datasetId,
        name: scanName.trim() || undefined,
        detector,
        layers: selectedLayers,
        threshold,
        seed,
      });
      onScanCreated(scan);
      setScanName('');
    } catch (err: any) {
      setError(err.message || 'Failed to start scan');
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="view-container">
      {/* Launch Configuration Form */}
      <div className="section-card scan-launcher-card">
        <div className="section-header">
          <div>
            <h3>Launch Multi-Layer Anomaly Detection Scan</h3>
            <span className="section-hint">
              Execute activation anomaly scoring across intermediate transformer layers
            </span>
          </div>
        </div>

        <form onSubmit={handleLaunch} className="scan-form">
          <div className="form-grid-3">
            <div className="form-group">
              <label className="form-label">Target Dataset</label>
              <select
                className="select-field"
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.version}) - {d.total_samples} samples
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Detector Engine</label>
              <select
                className="select-field"
                value={detector}
                onChange={(e) => setDetector(e.target.value as any)}
              >
                <option value="FLARE">FLARE (Layer-wise Activation Residuals)</option>
                <option value="ISOLATION_FOREST">Isolation Forest (Tree Anomaly)</option>
                <option value="KMEANS">K-Means (Centroid Distance Anomaly)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Scan Name (Optional)</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. Flare-Residual-AllLayers"
                value={scanName}
                onChange={(e) => setScanName(e.target.value)}
              />
            </div>
          </div>

          {/* Layer Selection Chips */}
          <div className="form-group mt-3">
            <label className="form-label">Embedding Layers to Inspect</label>
            <div className="layer-chips-group">
              {[0, 1, 2, 3, 4, 5].map((layerIdx) => {
                const isSelected = selectedLayers.includes(layerIdx);
                return (
                  <button
                    type="button"
                    key={layerIdx}
                    className={`layer-chip ${isSelected ? 'layer-chip-active' : ''}`}
                    onClick={() => toggleLayer(layerIdx)}
                  >
                    Layer {layerIdx}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="form-grid-3 mt-3">
            <div className="form-group">
              <label className="form-label">
                Decision Threshold: <span className="mono font-bold">{(threshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                className="slider-field"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Reproducibility Seed</label>
              <input
                type="number"
                className="input-field"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
              />
            </div>

            <div className="form-group align-end">
              <button type="submit" className="btn-primary w-full" disabled={launching}>
                {launching ? (
                  <>
                    <span className="spinner-sm" />
                    Starting Engine...
                  </>
                ) : (
                  'Start Anomaly Scan'
                )}
              </button>
            </div>
          </div>

          {error && <div className="alert-box alert-danger mt-3">{error}</div>}
        </form>
      </div>

      {/* Scans Execution History */}
      <div className="section-card mt-4">
        <div className="section-header">
          <div>
            <h3>Scan Execution History ({scans.length})</h3>
            <span className="section-hint">Track running background scans and review detection benchmarks</span>
          </div>
        </div>

        {scans.length === 0 ? (
          <div className="empty-state">No scans recorded. Launch a scan above to start anomaly profiling.</div>
        ) : (
          <div className="scans-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Scan Name</th>
                  <th>Detector</th>
                  <th>Status</th>
                  <th>AUROC</th>
                  <th>AUPRC</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Started At</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {scans.map((s) => {
                  const isRunning = s.status === 'RUNNING' || s.status === 'PENDING';
                  return (
                    <tr key={s.id}>
                      <td>
                        <strong>{s.name}</strong>
                        <div className="mono-sub">{s.id.slice(0, 8)}</div>
                      </td>
                      <td>
                        <span className="badge badge-secondary">{s.detector}</span>
                      </td>
                      <td>
                        <span className={`badge badge-${s.status === 'COMPLETED' ? 'success' : s.status === 'RUNNING' ? 'warning' : s.status === 'FAILED' ? 'danger' : 'secondary'}`}>
                          {isRunning && <span className="spinner-sm mr-1" />}
                          {s.status}
                        </span>
                      </td>
                      <td>{s.metrics?.auroc ? `${(s.metrics.auroc * 100).toFixed(1)}%` : '-'}</td>
                      <td>{s.metrics?.auprc ? `${(s.metrics.auprc * 100).toFixed(1)}%` : '-'}</td>
                      <td>{s.metrics?.precision ? `${(s.metrics.precision * 100).toFixed(1)}%` : '-'}</td>
                      <td>{s.metrics?.recall ? `${(s.metrics.recall * 100).toFixed(1)}%` : '-'}</td>
                      <td>{new Date(s.started_at).toLocaleTimeString()}</td>
                      <td>
                        {s.status === 'COMPLETED' ? (
                          <button
                            className="btn-link-sm"
                            onClick={() => onViewScanSamples(s.id)}
                          >
                            Suspicious Samples &rarr;
                          </button>
                        ) : (
                          <span className="text-muted-sm">Processing...</span>
                        )}
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
