import React, { useState, useEffect } from 'react';
import type { DatasetItem, ScanItem, PurificationResponse, PurificationPreview } from '../api';
import { triggerPurification, fetchPurificationPreview } from '../api';

interface PurificationViewProps {
  datasets: DatasetItem[];
  scans: ScanItem[];
  onPurificationComplete: (purified: PurificationResponse) => void;
  onNavigateToBenchmark: (rawDatasetId: string, purifiedDatasetId: string) => void;
}

export const PurificationView: React.FC<PurificationViewProps> = ({
  datasets,
  scans,
  onPurificationComplete,
  onNavigateToBenchmark,
}) => {
  const completedScans = scans.filter((s) => s.status === 'COMPLETED');
  const [datasetId, setDatasetId] = useState<string>(datasets[0]?.id || '');
  const [scanId, setScanId] = useState<string>(completedScans[0]?.id || '');
  const [riskThreshold, setRiskThreshold] = useState<number>(0.5);
  const [versionSuffix, setVersionSuffix] = useState<string>('purified-v1');
  const [purifying, setPurifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PurificationResponse | null>(null);
  const [preview, setPreview] = useState<PurificationPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);

  // Update preview directly from the backend whenever dataset, threshold, or scan changes
  useEffect(() => {
    if (!datasetId) {
      setPreview(null);
      return;
    }
    setPreviewLoading(true);
    let isCurrent = true;
    fetchPurificationPreview(datasetId, riskThreshold, scanId || undefined)
      .then((data) => {
        if (isCurrent) {
          setPreview(data);
          setPreviewLoading(false);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch preview:', err);
        if (isCurrent) setPreviewLoading(false);
      });

    return () => {
      isCurrent = false;
    };
  }, [datasetId, riskThreshold, scanId]);

  const selectedDataset = datasets.find((d) => d.id === datasetId);
  const totalSamples = preview?.total_original_samples ?? (selectedDataset?.total_samples || 0);
  const activeCount = preview?.projected_active_count ?? totalSamples;
  const quarantinedCount = preview?.projected_quarantined_count ?? 0;

  const handlePurify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetId) {
      setError('Please select a dataset to purify');
      return;
    }
    setPurifying(true);
    setError(null);

    try {
      const response = await triggerPurification({
        dataset_id: datasetId,
        experiment_id: scanId || undefined,
        risk_threshold: riskThreshold,
        version_suffix: versionSuffix.trim() || undefined,
      });
      setResult(response);
      onPurificationComplete(response);
    } catch (err: any) {
      setError(err.message || 'Purification failed');
    } finally {
      setPurifying(false);
    }
  };

  return (
    <div className="view-container">
      {/* Configuration Header Card */}
      <div className="section-card">
        <div className="section-header">
          <div>
            <h3>Dataset Purification & Automated Quarantine Station</h3>
            <span className="section-hint">
              Generate versioned, sanitized datasets by pruning high-risk backdoors and poison samples
            </span>
          </div>
        </div>

        <form onSubmit={handlePurify} className="purify-form">
          <div className="form-grid-2">
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
              <label className="form-label">Calibration Scan (Optional)</label>
              <select
                className="select-field"
                value={scanId}
                onChange={(e) => setScanId(e.target.value)}
              >
                <option value="">Use Latest Risk Baseline</option>
                {completedScans.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} ({s.detector})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-grid-2 mt-3">
            <div className="form-group">
              <label className="form-label">
                Risk Exclusion Threshold: <span className="mono font-bold">{(riskThreshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                className="slider-field"
                value={riskThreshold}
                onChange={(e) => setRiskThreshold(parseFloat(e.target.value))}
              />
              <span className="text-muted-sm">
                Samples with risk score ≥ {(riskThreshold * 100).toFixed(0)}% will be automatically quarantined.
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Purified Version Tag</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. purified-v1"
                value={versionSuffix}
                onChange={(e) => setVersionSuffix(e.target.value)}
              />
            </div>
          </div>

          {/* Real-Time Impact Projection */}
          <div className="projection-card mt-3">
            <div className="projection-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
              </svg>
              <span>Projected Sanitization Impact (Live Database Calculation)</span>
              {previewLoading && <span className="spinner-sm ml-2" />}
            </div>
            <div className="projection-stats">
              <div className="proj-stat">
                <span className="label">Total Original:</span>
                <strong>{totalSamples}</strong>
              </div>
              <div className="proj-stat text-success">
                <span className="label">Retained (Active):</span>
                <strong>{activeCount} {totalSamples > 0 ? `(${((activeCount / totalSamples) * 100).toFixed(0)}%)` : ''}</strong>
              </div>
              <div className="proj-stat text-danger">
                <span className="label">Quarantined (Excluded):</span>
                <strong>{quarantinedCount} {totalSamples > 0 ? `(${((quarantinedCount / totalSamples) * 100).toFixed(0)}%)` : ''}</strong>
              </div>
            </div>
          </div>

          <div className="form-actions mt-4">
            <button type="submit" className="btn-primary" disabled={purifying || datasets.length === 0}>
              {purifying ? (
                <>
                  <span className="spinner-sm" />
                  Sanitizing & Exporting Dataset...
                </>
              ) : (
                'Execute Dataset Purification & Export'
              )}
            </button>
          </div>

          {error && <div className="alert-box alert-danger mt-3">{error}</div>}
        </form>
      </div>

      {/* Purification Result Banner */}
      {result && (
        <div className="section-card result-card mt-4">
          <div className="result-header">
            <div className="success-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
            </div>
            <div>
              <h3>Dataset Successfully Purified & Exported!</h3>
              <p className="section-hint">
                New Dataset Version: <strong>{result.purified_dataset_version}</strong> (ID: {result.purified_dataset_id.slice(0, 8)})
              </p>
            </div>
          </div>

          <div className="metrics-grid mt-3">
            <div className="metric-card">
              <div className="metric-label">Original Samples</div>
              <div className="metric-value-lg">{result.total_original_samples}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Clean Samples Retained</div>
              <div className="metric-value-lg text-success">{result.active_count}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Quarantined Threats</div>
              <div className="metric-value-lg text-danger">{result.quarantined_count}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Artifact URI</div>
              <div className="metric-value mono-sub text-truncate" title={result.artifact_uri}>
                {result.artifact_uri.split('\\').pop()?.split('/').pop() || result.artifact_uri}
              </div>
            </div>
          </div>

          <div className="result-actions mt-4">
            <button
              className="btn-primary"
              onClick={() => onNavigateToBenchmark(result.original_dataset_id, result.purified_dataset_id)}
            >
              Benchmark Retraining (CA vs ASR) &rarr;
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
