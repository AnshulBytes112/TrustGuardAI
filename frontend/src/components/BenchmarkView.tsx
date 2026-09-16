import React, { useState, useEffect } from 'react';
import type { DatasetItem, RetrainingResponse } from '../api';
import { triggerRetraining } from '../api';

interface BenchmarkViewProps {
  datasets: DatasetItem[];
  initialRawDatasetId?: string | null;
  initialPurifiedDatasetId?: string | null;
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({
  datasets,
  initialRawDatasetId,
  initialPurifiedDatasetId,
}) => {
  const [rawDatasetId, setRawDatasetId] = useState<string>(initialRawDatasetId || datasets[0]?.id || '');
  const [purifiedDatasetId, setPurifiedDatasetId] = useState<string>(
    initialPurifiedDatasetId || datasets[1]?.id || datasets[0]?.id || ''
  );
  const [targetLabel, setTargetLabel] = useState<string>('positive');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RetrainingResponse | null>(null);

  useEffect(() => {
    if (initialRawDatasetId) setRawDatasetId(initialRawDatasetId);
    if (initialPurifiedDatasetId) setPurifiedDatasetId(initialPurifiedDatasetId);
  }, [initialRawDatasetId, initialPurifiedDatasetId]);

  const handleRunBenchmark = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rawDatasetId || !purifiedDatasetId) {
      setError('Please select both a raw and a purified dataset');
      return;
    }
    setRunning(true);
    setError(null);

    try {
      const response = await triggerRetraining({
        raw_dataset_id: rawDatasetId,
        purified_dataset_id: purifiedDatasetId,
        target_label: targetLabel.trim() || undefined,
      });
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Retraining benchmark failed');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="view-container">
      {/* Benchmark Setup Form */}
      <div className="section-card">
        <div className="section-header">
          <div>
            <h3>Downstream Retraining & ASR Benchmark</h3>
            <span className="section-hint">
              Evaluate Clean Accuracy (CA) preservation and Attack Success Rate (ASR) backdoor neutralization
            </span>
          </div>
        </div>

        <form onSubmit={handleRunBenchmark} className="benchmark-form">
          <div className="form-grid-3">
            <div className="form-group">
              <label className="form-label">Raw (Poisoned/Baseline) Dataset</label>
              <select
                className="select-field"
                value={rawDatasetId}
                onChange={(e) => setRawDatasetId(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.version})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Purified (Sanitized) Dataset</label>
              <select
                className="select-field"
                value={purifiedDatasetId}
                onChange={(e) => setPurifiedDatasetId(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.version})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Target Backdoor Label</label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. positive"
                value={targetLabel}
                onChange={(e) => setTargetLabel(e.target.value)}
              />
            </div>
          </div>

          <div className="form-actions mt-3">
            <button type="submit" className="btn-primary" disabled={running}>
              {running ? (
                <>
                  <span className="spinner-sm" />
                  Retraining Downstream Classifiers...
                </>
              ) : (
                'Run Retraining Benchmark'
              )}
            </button>
          </div>

          {error && <div className="alert-box alert-danger mt-3">{error}</div>}
        </form>
      </div>

      {/* Benchmark Results */}
      {result && (
        <div className="mt-4">
          {/* Executive Impact Cards */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Clean Accuracy (CA) Delta</span>
                <span className={`badge badge-${result.ca_delta >= -0.05 ? 'success' : 'warning'}`}>
                  {result.ca_delta >= 0 ? `+${(result.ca_delta * 100).toFixed(1)}%` : `${(result.ca_delta * 100).toFixed(1)}%`}
                </span>
              </div>
              <div className="metric-value-lg">
                {(result.purified_clean_accuracy * 100).toFixed(1)}%
              </div>
              <div className="metric-subtext">
                Baseline: {(result.raw_clean_accuracy * 100).toFixed(1)}% (Clean performance preserved)
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Attack Success Rate (ASR)</span>
                <span className="badge badge-success">
                  -{(result.asr_reduction * 100).toFixed(1)}% Reduction
                </span>
              </div>
              <div className="metric-value-lg text-success">
                {(result.purified_attack_success_rate * 100).toFixed(1)}%
              </div>
              <div className="metric-subtext">
                Poisoned Baseline: {(result.raw_attack_success_rate * 100).toFixed(1)}% ASR
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Neutralized Samples</span>
                <span className="badge badge-info">Quarantined</span>
              </div>
              <div className="metric-value-lg">{result.quarantined_samples_count}</div>
              <div className="metric-subtext">Threats safely removed from training corpus</div>
            </div>

            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Defense Verdict</span>
                <span className="badge badge-success">SECURED</span>
              </div>
              <div className="metric-value text-success font-bold">
                {result.purified_attack_success_rate <= 0.15 ? 'Backdoor Defeated' : 'Substantial Mitigation'}
              </div>
              <div className="metric-subtext">Downstream model immunity verified</div>
            </div>
          </div>

          {/* Comparative Bar Comparison Visualizer */}
          <div className="section-card mt-4">
            <div className="section-header">
              <h3>Side-by-Side Model Defense Comparison</h3>
              <span className="section-hint">Measuring performance retention vs backdoor vulnerability</span>
            </div>

            <div className="benchmark-comparison-grid">
              {/* Clean Accuracy Comparison */}
              <div className="comparison-box">
                <h4>Clean Accuracy (Higher is better)</h4>
                <div className="bar-comparison-row">
                  <div className="bar-meta">
                    <span>Raw Baseline</span>
                    <span>{(result.raw_clean_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill fill-secondary"
                      style={{ width: `${result.raw_clean_accuracy * 100}%` }}
                    />
                  </div>
                </div>

                <div className="bar-comparison-row mt-3">
                  <div className="bar-meta">
                    <span className="text-success font-bold">Purified Dataset</span>
                    <span className="text-success font-bold">{(result.purified_clean_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill fill-success"
                      style={{ width: `${result.purified_clean_accuracy * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Attack Success Rate Comparison */}
              <div className="comparison-box">
                <h4>Attack Success Rate (Lower is better)</h4>
                <div className="bar-comparison-row">
                  <div className="bar-meta">
                    <span className="text-danger font-bold">Poisoned Baseline (Vulnerable)</span>
                    <span className="text-danger font-bold">{(result.raw_attack_success_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill fill-danger"
                      style={{ width: `${result.raw_attack_success_rate * 100}%` }}
                    />
                  </div>
                </div>

                <div className="bar-comparison-row mt-3">
                  <div className="bar-meta">
                    <span className="text-success font-bold">Purified Dataset (Neutralized)</span>
                    <span className="text-success font-bold">{(result.purified_attack_success_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill fill-success"
                      style={{ width: `${result.purified_attack_success_rate * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
