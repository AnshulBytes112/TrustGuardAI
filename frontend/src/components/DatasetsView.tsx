import React, { useState } from 'react';
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
  onLaunchScan,
  onInspectSample,
}) => {
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(datasets[0]?.id || null);
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const loadDetail = async (id: string) => {
    setSelectedDatasetId(id);
    setLoadingDetail(true);
    try {
      const detail = await fetchDatasetDetail(id);
      setDatasetDetail(detail);
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setUploadError('Please choose a JSONL or JSON dataset file');
      return;
    }
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const uploaded = await uploadDatasetFile(selectedFile, datasetName.trim() || undefined);
      setUploadSuccess(`Successfully ingested dataset "${uploaded.name}" with ${uploaded.total_samples} samples.`);
      setSelectedFile(null);
      setDatasetName('');
      onRefresh();
      loadDetail(uploaded.id);
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="view-container">
      {/* Upload & Ingestion Card */}
      <div className="section-card upload-section">
        <div className="section-header">
          <div>
            <h3>Dataset Ingestion & Modality Catalog</h3>
            <span className="section-hint">Upload structured JSONL datasets or select existing benchmarks</span>
          </div>
        </div>

        <form onSubmit={handleFileUpload} className="upload-form">
          <div className="form-row">
            <div className="form-group flex-1">
              <label className="form-label">Dataset Name (Optional)</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. AG-News-Backdoor-Evaluation"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
              />
            </div>
            <div className="form-group flex-2">
              <label className="form-label">Upload JSONL File</label>
              <input
                type="file"
                accept=".jsonl,.json"
                className="file-input"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
            </div>
            <div className="form-group align-end">
              <button
                type="submit"
                className="btn-primary"
                disabled={uploading || !selectedFile}
              >
                {uploading ? (
                  <>
                    <span className="spinner-sm" />
                    Ingesting...
                  </>
                ) : (
                  'Ingest Dataset'
                )}
              </button>
            </div>
          </div>
          {uploadError && <div className="alert-box alert-danger mt-2">{uploadError}</div>}
          {uploadSuccess && <div className="alert-box alert-success mt-2">{uploadSuccess}</div>}
        </form>
      </div>

      {/* Dataset Grid & Detail Split */}
      <div className="dashboard-two-col mt-4">
        {/* Dataset Catalog Column */}
        <div className="dataset-catalog-col">
          <div className="section-header">
            <h3>Registered Datasets ({datasets.length})</h3>
          </div>
          <div className="dataset-card-list">
            {datasets.map((d) => {
              const isSelected = d.id === selectedDatasetId;
              const trainPct = d.total_samples > 0 ? (d.train_count / d.total_samples) * 100 : 0;
              const valPct = d.total_samples > 0 ? (d.val_count / d.total_samples) * 100 : 0;
              const testPct = d.total_samples > 0 ? (d.test_count / d.total_samples) * 100 : 0;

              return (
                <div
                  key={d.id}
                  className={`dataset-card ${isSelected ? 'dataset-card-selected' : ''}`}
                  onClick={() => loadDetail(d.id)}
                >
                  <div className="card-top">
                    <span className="dataset-title">{d.name}</span>
                    <span className="badge badge-info">{d.version}</span>
                  </div>
                  <div className="card-stats">
                    <div className="stat-pill">
                      <span>Total:</span> <strong>{d.total_samples}</strong>
                    </div>
                    <div className="stat-pill">
                      <span>Modality:</span> <strong>{d.modality}</strong>
                    </div>
                    <div className="stat-pill">
                      <span>Source:</span> <strong>{d.source}</strong>
                    </div>
                  </div>

                  {/* Split Distribution Bar */}
                  <div className="split-progress-container" title={`Train: ${d.train_count}, Val: ${d.val_count}, Test: ${d.test_count}`}>
                    <div className="split-seg split-train" style={{ width: `${trainPct}%` }} />
                    <div className="split-seg split-val" style={{ width: `${valPct}%` }} />
                    <div className="split-seg split-test" style={{ width: `${testPct}%` }} />
                  </div>
                  <div className="split-legend">
                    <span className="dot dot-train">Train ({d.train_count})</span>
                    <span className="dot dot-val">Val ({d.val_count})</span>
                    <span className="dot dot-test">Test ({d.test_count})</span>
                  </div>

                  <div className="card-actions mt-3">
                    <button
                      className="btn-secondary btn-sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onLaunchScan(d.id);
                      }}
                    >
                      Scan Anomaly &rarr;
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dataset Detail & Samples Preview Column */}
        <div className="dataset-detail-col">
          <div className="section-card">
            <div className="section-header">
              <h3>
                {datasetDetail ? `Samples in "${datasetDetail.name}"` : 'Dataset Sample Browser'}
              </h3>
              {datasetDetail && (
                <span className="badge badge-secondary">{datasetDetail.samples.length} Samples</span>
              )}
            </div>

            {loadingDetail ? (
              <div className="loading-state">
                <div className="spinner" />
                <p>Loading samples...</p>
              </div>
            ) : datasetDetail ? (
              <div className="samples-preview-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Sample ID</th>
                      <th>Text Snippet</th>
                      <th>Split</th>
                      <th>Label</th>
                      <th>State</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetDetail.samples.slice(0, 15).map((s) => (
                      <tr key={s.id}>
                        <td className="mono-sub">{s.external_sample_id || s.id.slice(0, 8)}</td>
                        <td className="text-snippet-cell" title={s.text}>
                          {s.text.length > 70 ? s.text.slice(0, 70) + '...' : s.text}
                        </td>
                        <td>
                          <span className="badge badge-secondary">{s.split}</span>
                        </td>
                        <td>{s.label || 'None'}</td>
                        <td>
                          <span className={`badge badge-${s.state === 'QUARANTINED' ? 'danger' : s.state === 'RESTORED' ? 'info' : 'success'}`}>
                            {s.state}
                          </span>
                        </td>
                        <td>
                          <button
                            className="btn-link-sm"
                            onClick={() => onInspectSample(s.id)}
                          >
                            Inspect XAI
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                Select a dataset on the left to inspect its samples and distribution.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
