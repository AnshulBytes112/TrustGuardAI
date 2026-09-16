import React, { useState, useEffect } from 'react';
import type { DatasetItem, PurificationResponse, PurificationPreview } from '../api';
import { triggerPurification, fetchPurificationPreview } from '../api';

interface PurificationViewProps {
  datasets: DatasetItem[];
  onPurificationComplete: (purified: PurificationResponse) => void;
  onNavigateToBenchmark: (rawDatasetId: string, purifiedDatasetId: string) => void;
}

export const PurificationView: React.FC<PurificationViewProps> = ({
  datasets,
  onPurificationComplete,
  onNavigateToBenchmark,
}) => {
  const [method, setMethod] = useState<'remove' | 'reweight' | 'finegrained'>('remove');
  const [threshold, setThreshold] = useState<number>(0.60);
  const [selectedDsId, setSelectedDsId] = useState<string>(datasets[0]?.id || '');
  const [preview, setPreview] = useState<PurificationPreview | null>(null);
  const [purifying, setPurifying] = useState(false);
  const [result, setResult] = useState<PurificationResponse | null>(null);

  useEffect(() => {
    if (datasets.length > 0 && !selectedDsId) {
      setSelectedDsId(datasets[0].id);
    }
  }, [datasets, selectedDsId]);

  useEffect(() => {
    if (!selectedDsId) return;
    fetchPurificationPreview(selectedDsId, threshold)
      .then(setPreview)
      .catch((err) => console.error(err));
  }, [selectedDsId, threshold]);

  const currentDs = datasets.find((d) => d.id === selectedDsId) || datasets[0];
  const originalCount = preview?.total_original_samples ?? (currentDs?.total_samples || 100);
  const toRemoveCount = preview?.projected_quarantined_count ?? 20;
  const remainingCount = preview?.projected_active_count ?? (originalCount - toRemoveCount);

  const removePct = originalCount > 0 ? ((toRemoveCount / originalCount) * 100).toFixed(1) : '20.0';
  const remainPct = originalCount > 0 ? ((remainingCount / originalCount) * 100).toFixed(1) : '80.0';

  const handleExecute = async () => {
    if (!selectedDsId) {
      alert('Please select a dataset to purify.');
      return;
    }
    setPurifying(true);
    try {
      const res = await triggerPurification({
        dataset_id: selectedDsId,
        risk_threshold: threshold,
        version_suffix: 'clean',
      });
      setResult(res);
      onPurificationComplete(res);
    } catch (err: any) {
      alert(`Purification failed: ${err.message}`);
    } finally {
      setPurifying(false);
    }
  };

  return (
    <div className="view-container">
      {/* Header */}
      <div className="view-header">
        <div className="view-header-left">
          <div className="view-tag">MITIGATION</div>
          <h1>Purification Studio</h1>
          <p>
            Remove or mitigate poisoned samples from your training data.
            Apply data purification techniques to create cleaner datasets.
          </p>
        </div>
        <div className="view-header-quote">
          "Cleaner data.<br />Stronger models.<br />A safer tomorrow."
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="grid-2-equal">
        {/* Left: Purification Method */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Purification Method</div>
          </div>

          <div className="radio-card-group">
            <div
              className={`radio-card ${method === 'remove' ? 'active' : ''}`}
              onClick={() => setMethod('remove')}
            >
              <input
                type="radio"
                name="method"
                checked={method === 'remove'}
                onChange={() => setMethod('remove')}
                style={{ accentColor: 'var(--primary)', marginTop: '2px' }}
              />
              <div className="radio-card-content">
                <h4>Remove Suspicious Samples</h4>
                <p>Filter out high-anomaly samples</p>
              </div>
            </div>

            <div
              className={`radio-card ${method === 'reweight' ? 'active' : ''}`}
              onClick={() => setMethod('reweight')}
            >
              <input
                type="radio"
                name="method"
                checked={method === 'reweight'}
                onChange={() => setMethod('reweight')}
                style={{ accentColor: 'var(--primary)', marginTop: '2px' }}
              />
              <div className="radio-card-content">
                <h4>Reweight Samples</h4>
                <p>Down-weight suspicious samples</p>
              </div>
            </div>

            <div
              className={`radio-card ${method === 'finegrained' ? 'active' : ''}`}
              onClick={() => setMethod('finegrained')}
            >
              <input
                type="radio"
                name="method"
                checked={method === 'finegrained'}
                onChange={() => setMethod('finegrained')}
                style={{ accentColor: 'var(--primary)', marginTop: '2px' }}
              />
              <div className="radio-card-content">
                <h4>Fine-grained Filtering</h4>
                <p>Confidence-based filtering</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Purification Settings */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Purification Settings</div>
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <label className="form-label" style={{ margin: 0 }}>Anomaly Score Threshold</label>
              <span className="mono" style={{
                background: 'var(--bg-card-subtle)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                padding: '0.2rem 0.5rem',
                fontSize: '0.82rem',
                fontWeight: '600',
              }}>
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--primary)' }}
            />
          </div>

          {/* Preview Impact Box */}
          <div style={{ marginTop: '1.25rem' }}>
            <div className="card-subtitle" style={{ fontWeight: '700', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              Preview Impact
            </div>
            <div className="impact-preview-grid">
              <div className="impact-box neutral">
                <div className="impact-box-lbl" style={{ color: 'var(--text-muted)' }}>Original Samples</div>
                <div className="impact-box-val" style={{ color: 'var(--text-primary)' }}>{originalCount}</div>
              </div>

              <div className="impact-box danger">
                <div className="impact-box-lbl">To Remove</div>
                <div className="impact-box-val">{toRemoveCount} ({removePct}%)</div>
              </div>

              <div className="impact-box success">
                <div className="impact-box-lbl">Remaining</div>
                <div className="impact-box-val">{remainingCount} ({remainPct}%)</div>
              </div>
            </div>
          </div>

          <button
            className="btn-forest"
            style={{ width: '100%', justifyContent: 'center', marginTop: '1rem' }}
            onClick={handleExecute}
            disabled={purifying || datasets.length === 0}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            {purifying ? 'Purifying Dataset...' : 'Preview Purification'}
          </button>
        </div>
      </div>

      {/* Success Notification Card */}
      {result && (
        <div className="card" style={{ marginTop: '1.5rem', borderColor: 'var(--success-border)', background: 'var(--success-bg)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h4 style={{ color: 'var(--success-text)', fontWeight: '700' }}>Dataset Successfully Purified!</h4>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                New version created: <strong>{result.purified_dataset_version}</strong> ({result.active_count} clean samples retained, {result.quarantined_count} threats quarantined).
              </p>
            </div>
            <button
              className="btn-forest"
              onClick={() => onNavigateToBenchmark(result.original_dataset_id, result.purified_dataset_id)}
            >
              Retraining Benchmark &rarr;
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
