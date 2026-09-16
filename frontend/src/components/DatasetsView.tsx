import React, { useState, useEffect } from 'react';
import type { DatasetItem, DatasetDetail } from '../api';
import { fetchDatasetDetail, uploadDatasetFile } from '../api';

interface DatasetsViewProps {
  datasets: DatasetItem[];
  onRefresh: () => void;
  onLaunchScan: (datasetId: string) => void;
  onInspectSample: (sampleId: string) => void;
}

export const DatasetsView: React.FC<DatasetsViewProps> = ({
  datasets,
  onRefresh,
  onInspectSample,
}) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(datasets[0]?.id || null);
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(null);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    if (datasets.length > 0 && !selectedDatasetId) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [datasets, selectedDatasetId]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    fetchDatasetDetail(selectedDatasetId)
      .then(setDatasetDetail)
      .catch((err) => console.error(err));
  }, [selectedDatasetId]);

  const currentDs = datasets.find((d) => d.id === selectedDatasetId) || datasets[0];

  const totalSamples = currentDs ? currentDs.total_samples : 100;
  const trainCount = currentDs ? currentDs.train_count : 60;
  const valCount = currentDs ? currentDs.val_count : 20;
  const testCount = currentDs ? currentDs.test_count : 20;
  const labelMode = currentDs ? currentDs.label_mode.replace('_', ' ') : 'Fully Labelled';

  const trainPct = totalSamples > 0 ? Math.round((trainCount / totalSamples) * 100) : 60;
  const valPct = totalSamples > 0 ? Math.round((valCount / totalSamples) * 100) : 20;
  const testPct = totalSamples > 0 ? Math.round((testCount / totalSamples) * 100) : 20;

  // Compute label counts from samples preview
  const samplesList = datasetDetail?.samples || [];
  const posCount = samplesList.filter((s) => s.label && s.label.toLowerCase().includes('pos')).length || Math.round(totalSamples * 0.5);
  const negCount = samplesList.filter((s) => s.label && s.label.toLowerCase().includes('neg')).length || (totalSamples - posCount);

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setUploadError('Please choose a file');
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const created = await uploadDatasetFile(uploadFile, uploadName.trim() || undefined);
      setUploadModalOpen(false);
      setUploadFile(null);
      setUploadName('');
      onRefresh();
      setSelectedDatasetId(created.id);
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="view-container">
      {/* View Header */}
      <div className="view-header">
        <div className="view-header-left">
          <div className="view-tag">DATA</div>
          <h1>Dataset Studio</h1>
          <p>
            Upload, validate, and explore your dataset.
            Use JSONL/CSV files with text, label (optional), and split fields.
          </p>
        </div>
        <button className="btn-forest" onClick={() => setUploadModalOpen(true)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          Upload Dataset
        </button>
      </div>

      {/* Top Stat Row (5 Metric Cards) */}
      <div className="grid-5">
        <div className="metric-box">
          <div className="metric-data">
            <div className="metric-lbl">Total Samples</div>
            <div className="metric-val">{totalSamples}</div>
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-data">
            <div className="metric-lbl" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="legend-dot dot-train" /> Train
            </div>
            <div className="metric-val">{trainCount} ({trainPct}%)</div>
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-data">
            <div className="metric-lbl" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="legend-dot dot-val" /> Validation
            </div>
            <div className="metric-val">{valCount} ({valPct}%)</div>
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-data">
            <div className="metric-lbl" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="legend-dot dot-test" /> Test
            </div>
            <div className="metric-val">{testCount} ({testPct}%)</div>
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-data">
            <div className="metric-lbl">🏷️ Label Mode</div>
            <div className="metric-val" style={{ fontSize: '1.05rem', textTransform: 'capitalize' }}>
              {labelMode.toLowerCase()}
            </div>
          </div>
        </div>
      </div>

      {/* Two Columns Layout */}
      <div className="grid-2">
        {/* Left: Dataset Preview */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Dataset Preview</div>
            <button className="btn-link-sm">View all &rarr;</button>
          </div>
          <div className="table-container">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Text</th>
                  <th>Label</th>
                  <th>Split</th>
                </tr>
              </thead>
              <tbody>
                {samplesList.length === 0 ? (
                  <>
                    <tr>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_001</td>
                      <td>The product quality is excellent and delivery was fast...</td>
                      <td><span className="pill pill-green">positive</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                    <tr>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_002</td>
                      <td>Very disappointing quality. The product stopped working...</td>
                      <td><span className="pill pill-red">negative</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                    <tr>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_003</td>
                      <td>Great value for money. Highly recommended for daily use.</td>
                      <td><span className="pill pill-green">positive</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                    <tr>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_004</td>
                      <td>Poor build quality. Not worth the price paid at all.</td>
                      <td><span className="pill pill-red">negative</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                    <tr>
                      <td className="mono" style={{ fontSize: '0.78rem' }}>sample_005</td>
                      <td>Easy to use and works exactly as expected without issues.</td>
                      <td><span className="pill pill-green">positive</span></td>
                      <td><span className="pill pill-blue">train</span></td>
                    </tr>
                  </>
                ) : (
                  samplesList.slice(0, 6).map((s) => (
                    <tr
                      key={s.id}
                      onClick={() => onInspectSample(s.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td className="mono" style={{ fontSize: '0.78rem' }}>{s.external_sample_id || s.id.slice(0, 8)}</td>
                      <td title={s.text}>
                        {s.text.length > 55 ? s.text.slice(0, 55) + '...' : s.text}
                      </td>
                      <td>
                        <span className={`pill ${s.label && s.label.toLowerCase().includes('pos') ? 'pill-green' : s.label && s.label.toLowerCase().includes('neg') ? 'pill-red' : 'pill-gray'}`}>
                          {s.label || 'none'}
                        </span>
                      </td>
                      <td>
                        <span className="pill pill-blue">{s.split.toLowerCase()}</span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Charts Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Split Distribution */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Split Distribution</div>
            </div>
            <div className="donut-chart-container">
              <div className="donut-svg-wrap">
                <svg width="110" height="110" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="45" fill="none" stroke="#e2e8e2" strokeWidth="14" />
                  <circle
                    cx="60"
                    cy="60"
                    r="45"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="14"
                    strokeDasharray={`${(trainPct / 100) * 282.7} 282.7`}
                    strokeDashoffset="0"
                    transform="rotate(-90 60 60)"
                  />
                  <circle
                    cx="60"
                    cy="60"
                    r="45"
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="14"
                    strokeDasharray={`${(valPct / 100) * 282.7} 282.7`}
                    strokeDashoffset={`-${(trainPct / 100) * 282.7}`}
                    transform="rotate(-90 60 60)"
                  />
                  <circle
                    cx="60"
                    cy="60"
                    r="45"
                    fill="none"
                    stroke="#8b5cf6"
                    strokeWidth="14"
                    strokeDasharray={`${(testPct / 100) * 282.7} 282.7`}
                    strokeDashoffset={`-${((trainPct + valPct) / 100) * 282.7}`}
                    transform="rotate(-90 60 60)"
                  />
                </svg>
                <div className="donut-center-text">
                  <div className="donut-center-val">{totalSamples}</div>
                  <div className="donut-center-lbl">Samples</div>
                </div>
              </div>

              <div className="chart-legend">
                <div className="legend-item">
                  <span className="legend-dot dot-train" />
                  <span>Train: <strong>{trainCount} ({trainPct}%)</strong></span>
                </div>
                <div className="legend-item">
                  <span className="legend-dot dot-val" />
                  <span>Validation: <strong>{valCount} ({valPct}%)</strong></span>
                </div>
                <div className="legend-item">
                  <span className="legend-dot dot-test" />
                  <span>Test: <strong>{testCount} ({testPct}%)</strong></span>
                </div>
              </div>
            </div>
          </div>

          {/* Label Distribution */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Label Distribution</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: '2.5rem', height: '110px', paddingTop: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{
                  width: '42px',
                  height: '60px',
                  backgroundColor: '#10b981',
                  borderRadius: '6px 6px 0 0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontWeight: '700',
                  fontSize: '0.8rem',
                }}>
                  {posCount}
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-secondary)' }}>Positive</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{
                  width: '42px',
                  height: '60px',
                  backgroundColor: '#ef4444',
                  borderRadius: '6px 6px 0 0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontWeight: '700',
                  fontSize: '0.8rem',
                }}>
                  {negCount}
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-secondary)' }}>Negative</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {uploadModalOpen && (
        <div className="modal-overlay" onClick={() => setUploadModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '480px' }}>
            <div className="card-header">
              <div className="card-title">Upload Dataset (.jsonl)</div>
              <button className="btn-link-sm" onClick={() => setUploadModalOpen(false)}>✕</button>
            </div>
            <form onSubmit={handleUploadSubmit}>
              <div className="form-group">
                <label className="form-label">Dataset Name (Optional)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. trustguard_demo_100.jsonl"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">JSONL File</label>
                <input
                  type="file"
                  accept=".jsonl,.json"
                  className="form-input"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
              </div>
              {uploadError && <div style={{ color: 'var(--danger)', fontSize: '0.82rem', marginBottom: '0.75rem' }}>{uploadError}</div>}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.25rem' }}>
                <button type="button" className="btn-outline" onClick={() => setUploadModalOpen(false)}>Cancel</button>
                <button type="submit" className="btn-forest" disabled={uploading || !uploadFile}>
                  {uploading ? 'Uploading...' : 'Upload & Process'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
