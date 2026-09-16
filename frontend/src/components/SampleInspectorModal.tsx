import React, { useState, useEffect } from 'react';
import type { SampleInvestigation } from '../api';
import {
  fetchSampleInvestigation,
  quarantineSample,
  restoreSample,
} from '../api';

interface SampleInspectorModalProps {
  sampleId: string | null;
  onClose: () => void;
  onSampleUpdated?: () => void;
}

export const SampleInspectorModal: React.FC<SampleInspectorModalProps> = ({
  sampleId,
  onClose,
  onSampleUpdated,
}) => {
  const [data, setData] = useState<SampleInvestigation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reason, setReason] = useState('');
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    if (!sampleId) return;
    setLoading(true);
    setError(null);
    fetchSampleInvestigation(sampleId)
      .then(setData)
      .catch((err) => setError(err.message || 'Failed to load sample investigation'))
      .finally(() => setLoading(false));
  }, [sampleId]);

  if (!sampleId) return null;

  const handleQuarantine = async () => {
    if (!reason.trim()) {
      alert('Please provide a reason for quarantining this sample.');
      return;
    }
    setActionLoading(true);
    try {
      const updated = await quarantineSample(sampleId, reason);
      setData(updated);
      setReason('');
      if (onSampleUpdated) onSampleUpdated();
    } catch (err: any) {
      alert(`Quarantine failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleRestore = async () => {
    if (!reason.trim()) {
      alert('Please provide a reason for restoring this sample.');
      return;
    }
    setActionLoading(true);
    try {
      const updated = await restoreSample(sampleId, reason);
      setData(updated);
      setReason('');
      if (onSampleUpdated) onSampleUpdated();
    } catch (err: any) {
      alert(`Restore failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const renderTrajectoryBadge = (trajectory?: string) => {
    switch (trajectory) {
      case 'early':
        return <span className="badge badge-warning">Trajectory: Early Layers (Feature-Level)</span>;
      case 'middle':
        return <span className="badge badge-info">Trajectory: Middle Layers (Semantic Shift)</span>;
      case 'late':
        return <span className="badge badge-danger">Trajectory: Late Layers (Classification Trigger)</span>;
      case 'uniform':
        return <span className="badge badge-secondary">Trajectory: Uniform Anomaly</span>;
      default:
        return <span className="badge badge-secondary">Trajectory: {trajectory || 'N/A'}</span>;
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-lg" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Sample Deep Investigation & XAI</h2>
            <div className="modal-subtitle">
              External ID: <span className="mono">{data?.external_sample_id || sampleId}</span>
            </div>
          </div>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        {loading ? (
          <div className="loading-state">
            <div className="spinner" />
            <p>Analyzing activation patterns & token attributions...</p>
          </div>
        ) : error ? (
          <div className="error-card">
            <h4>Error Loading Investigation</h4>
            <p>{error}</p>
          </div>
        ) : data ? (
          <div className="modal-body">
            {/* Risk & State Meta Bar */}
            <div className="meta-card-grid">
              <div className="metric-card">
                <div className="metric-label">Composite Risk</div>
                <div className="metric-value-lg">
                  {(data.risk_score * 100).toFixed(1)}%
                </div>
                <span className={`badge badge-${data.risk_level === 'HIGH' ? 'danger' : data.risk_level === 'MEDIUM' ? 'warning' : 'success'}`}>
                  {data.risk_level} RISK
                </span>
              </div>

              <div className="metric-card">
                <div className="metric-label">Sample State</div>
                <div className="metric-value">
                  <span className={`badge badge-${data.state === 'QUARANTINED' ? 'danger' : data.state === 'RESTORED' ? 'info' : 'success'}`}>
                    {data.state}
                  </span>
                </div>
                <div className="metric-subtext">Split: {data.split.toUpperCase()} | Label: {data.label || 'None'}</div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Dominant Layer</div>
                <div className="metric-value">Layer {data.dominant_layer}</div>
                <div className="metric-subtext">
                  {renderTrajectoryBadge(data.trajectory)}
                </div>
              </div>
            </div>

            {/* Evidence Synthesis Alert */}
            <div className="alert-box alert-info">
              <div className="alert-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="16" x2="12" y2="12"/>
                  <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                Natural Language Evidence Synthesis
              </div>
              <p className="alert-desc">{data.evidence_summary || data.dominant_evidence || 'No anomaly evidence detected.'}</p>
            </div>

            {/* Token Saliency Heatmap */}
            <div className="section-card">
              <div className="section-header">
                <h3>Token Attribution Heatmap</h3>
                <span className="section-hint">Highlighted spans indicate high saliency anomaly triggers</span>
              </div>
              <div className="token-heatmap-container">
                {data.token_attributions && data.token_attributions.length > 0 ? (
                  data.token_attributions.map((tokenObj, idx) => {
                    const saliency = Math.min(1.0, Math.max(0, tokenObj.saliency_score));
                    const isSuspicious = tokenObj.is_suspicious_span;
                    const bgAlpha = (saliency * 0.5 + (isSuspicious ? 0.25 : 0.05)).toFixed(2);
                    const bgColor = isSuspicious 
                      ? `rgba(239, 68, 68, ${bgAlpha})` 
                      : `rgba(99, 102, 241, ${bgAlpha})`;
                    const borderColor = isSuspicious ? 'rgba(239, 68, 68, 0.7)' : 'transparent';

                    return (
                      <span
                        key={idx}
                        className={`heatmap-token ${isSuspicious ? 'token-suspicious' : ''}`}
                        style={{
                          backgroundColor: bgColor,
                          borderBottom: isSuspicious ? `2px solid ${borderColor}` : 'none',
                        }}
                        title={`Token: "${tokenObj.token}" | Saliency: ${(saliency * 100).toFixed(1)}% | ${isSuspicious ? 'Suspicious Span' : 'Normal'}`}
                      >
                        {tokenObj.token}
                      </span>
                    );
                  })
                ) : (
                  <p className="plain-text-sample">{data.text}</p>
                )}
              </div>
            </div>

            {/* Layer Anomaly Decomposition */}
            {data.layer_scores && Object.keys(data.layer_scores).length > 0 && (
              <div className="section-card">
                <div className="section-header">
                  <h3>Multi-Layer Anomaly Decomposition</h3>
                  <span className="section-hint">Layer-by-layer residual deviation metrics</span>
                </div>
                <div className="layer-bars-container">
                  {Object.entries(data.layer_scores).map(([layerName, score]) => {
                    const attribution = data.layer_attributions?.[layerName] ?? score;
                    const pct = Math.min(100, Math.max(0, attribution * 100));
                    const isDominant = layerName.includes(String(data.dominant_layer));
                    return (
                      <div key={layerName} className={`layer-bar-row ${isDominant ? 'dominant-row' : ''}`}>
                        <div className="layer-label">
                          <span>{layerName}</span>
                          {isDominant && <span className="dominant-tag">DOMINANT</span>}
                        </div>
                        <div className="progress-track">
                          <div
                            className={`progress-fill ${isDominant ? 'fill-dominant' : ''}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <div className="layer-value">{(score * 100).toFixed(1)}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Audit History Log */}
            {data.quarantine_history && data.quarantine_history.length > 0 && (
              <div className="section-card">
                <div className="section-header">
                  <h3>Audit Trail & Quarantine Log</h3>
                </div>
                <div className="audit-log-list">
                  {data.quarantine_history.map((item) => (
                    <div key={item.id} className="audit-item">
                      <div className="audit-meta">
                        <span className={`badge badge-${item.action === 'QUARANTINE' ? 'danger' : 'success'}`}>
                          {item.action}
                        </span>
                        <span className="audit-timestamp">{new Date(item.timestamp).toLocaleString()}</span>
                      </div>
                      <div className="audit-reason">
                        <strong>Reason:</strong> {item.reason}
                      </div>
                      <div className="audit-transition">
                        State: {item.previous_state} &rarr; {item.new_state}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quarantine / Restore Actions */}
            <div className="action-card-footer">
              <div className="action-input-row">
                <input
                  type="text"
                  className="input-field"
                  placeholder="Enter audit rationale/reason..."
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
                {data.state === 'QUARANTINED' ? (
                  <button
                    className="btn-success"
                    onClick={handleRestore}
                    disabled={actionLoading || !reason.trim()}
                  >
                    {actionLoading ? 'Restoring...' : 'Restore Sample to Active'}
                  </button>
                ) : (
                  <button
                    className="btn-danger"
                    onClick={handleQuarantine}
                    disabled={actionLoading || !reason.trim()}
                  >
                    {actionLoading ? 'Quarantining...' : 'Quarantine Sample'}
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};
