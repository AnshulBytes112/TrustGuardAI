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
  initialDatasetId,
  onScanCreated,
  onViewScanSamples,
}) => {
  const [activeTab, setActiveTab] = useState<'dataset' | 'single'>('dataset');
  const [selectedDsId, setSelectedDsId] = useState<string>(initialDatasetId || datasets[0]?.id || '');
  const [detector, setDetector] = useState<'FLARE' | 'ISOLATION_FOREST' | 'KMEANS'>('FLARE');
  const [repModel, setRepModel] = useState<string>('DistilBERT');
  const [singleText, setSingleText] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDsId) {
      setError('Please select a dataset');
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const scan = await createScan({
        dataset_id: selectedDsId,
        detector,
        name: `${detector.toLowerCase()}_scan_${new Date().toISOString().slice(11, 19)}`,
      });
      onScanCreated(scan);
      onViewScanSamples(scan.id);
    } catch (err: any) {
      setError(err.message || 'Failed to start scan');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="view-container">
      {/* Header */}
      <div className="view-header">
        <div className="view-header-left">
          <div className="view-tag">DETECTION</div>
          <h1>Anomaly Scan Center</h1>
          <p>
            Run anomaly detection to identify suspicious training samples.
            Use FLARE to detect outliers in representation space.
          </p>
        </div>
        <button
          className="btn-forest"
          onClick={handleLaunch}
          disabled={running || datasets.length === 0}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          {running ? 'Running...' : 'Run Scan'}
        </button>
      </div>

      {/* Tabs */}
      <div className="tab-pill-group">
        <button
          className={`tab-pill ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataset')}
        >
          Dataset Scan
        </button>
        <button
          className={`tab-pill ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          Single Text Scan
        </button>
      </div>

      {activeTab === 'dataset' ? (
        <div className="grid-2-equal">
          {/* Left: Scan Configuration */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Scan Configuration</div>
            </div>

            <form onSubmit={handleLaunch}>
              <div className="form-group">
                <label className="form-label">Select Dataset</label>
                <select
                  className="form-select"
                  value={selectedDsId}
                  onChange={(e) => setSelectedDsId(e.target.value)}
                >
                  {datasets.length === 0 ? (
                    <option value="">trustguard_demo_100.jsonl</option>
                  ) : (
                    datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.total_samples} samples)
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Detection Model</label>
                <select
                  className="form-select"
                  value={detector}
                  onChange={(e) => setDetector(e.target.value as any)}
                >
                  <option value="FLARE">FLARE (Default)</option>
                  <option value="ISOLATION_FOREST">Isolation Forest</option>
                  <option value="KMEANS">K-Means Centroid Distance</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Representation Model</label>
                <select
                  className="form-select"
                  value={repModel}
                  onChange={(e) => setRepModel(e.target.value)}
                >
                  <option value="DistilBERT">DistilBERT</option>
                  <option value="RoBERTa">RoBERTa (Coming Soon)</option>
                </select>
              </div>

              {error && <div style={{ color: 'var(--danger)', fontSize: '0.82rem', marginBottom: '0.75rem' }}>{error}</div>}

              <button
                type="submit"
                className="btn-forest"
                style={{ width: '100%', justifyContent: 'center', marginTop: '0.5rem' }}
                disabled={running}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                {running ? 'Executing Multi-Layer Scan...' : 'Run Anomaly Scan'}
              </button>
            </form>
          </div>

          {/* Right: Model Settings */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Model Settings</div>
            </div>

            <div className="kv-row">
              <span className="kv-key">🔤 Model</span>
              <span className="kv-val">DistilBERT (multi-layer)</span>
            </div>

            <div className="kv-row">
              <span className="kv-key">⚙️ Pooling</span>
              <span className="kv-val">Mean Pooling</span>
            </div>

            <div className="kv-row">
              <span className="kv-key">🥞 Layers</span>
              <span className="kv-val">[2, 4, 6]</span>
            </div>

            <div className="kv-row">
              <span className="kv-key">🎯 Threshold</span>
              <span className="kv-val">Auto (Youden's J)</span>
            </div>

            <div className="kv-row">
              <span className="kv-key">📦 Batch Size</span>
              <span className="kv-val">32</span>
            </div>

            <div className="kv-row">
              <span className="kv-key">💻 Device</span>
              <span className="kv-val">Auto (CUDA if available)</span>
            </div>
          </div>
        </div>
      ) : (
        /* Single Text Scan View */
        <div className="card">
          <div className="card-header">
            <div className="card-title">Single Text Anomaly Inspection</div>
          </div>
          <div className="form-group">
            <label className="form-label">Input Text to Inspect</label>
            <textarea
              className="form-input"
              rows={4}
              placeholder="Paste arbitrary input text to analyze embedding layer anomalies..."
              value={singleText}
              onChange={(e) => setSingleText(e.target.value)}
            />
          </div>
          <button className="btn-forest" disabled={!singleText.trim()}>
            Inspect Text
          </button>
        </div>
      )}
    </div>
  );
};
