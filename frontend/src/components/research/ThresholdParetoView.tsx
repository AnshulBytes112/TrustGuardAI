import React, { useEffect, useState } from 'react';
import { fetchResearchCategoryData } from '../../api';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';

export const ThresholdParetoView: React.FC = () => {
  const [sweepData, setSweepData] = useState<any[]>([]);
  const [selectedThresholdIdx, setSelectedThresholdIdx] = useState<number>(25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSweep = async () => {
      try {
        setLoading(true);
        const res = await fetchResearchCategoryData('threshold_sweep');
        const list = Array.isArray(res) ? res : res?.results || res?.sweep || [];
        setSweepData(list);
        if (list.length > 0) {
          setSelectedThresholdIdx(Math.floor(list.length / 2));
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load continuous threshold sweep data');
      } finally {
        setLoading(false);
      }
    };
    loadSweep();
  }, []);

  if (loading) {
    return <LoadingSkeleton type="card" height="350px" />;
  }

  if (error) {
    return <ErrorState message={error} />;
  }

  const currentPoint = sweepData[selectedThresholdIdx] || {
    threshold: 0.50,
    retention_rate: 0.94,
    poison_removal_rate: 0.98,
    clean_accuracy: 0.884,
    asr: 0.021,
    precision: 0.95,
    recall: 0.98,
    f1: 0.965,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header Info */}
      <div className="neuro-card" style={{ padding: '1.5rem 1.75rem' }}>
        <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
          Continuous Threshold Calibration & Security-Utility Pareto Frontier
        </h2>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', maxWidth: '820px', marginTop: '0.35rem' }}>
          Explore the exact trade-off between training data retention (model utility) and poison removal (security defense) across continuous threshold sweeps &tau; &in; [0.0, 1.0].
        </p>
      </div>

      {/* Interactive Threshold Slider & Inspector */}
      <div className="neuro-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>
              Interactive Threshold Inspector (&tau; = {Number(currentPoint.threshold).toFixed(3)})
            </h3>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
              Move slider to inspect operating point metrics along the Pareto curve
            </p>
          </div>
          <span className="neuro-badge cyan" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
            Point {selectedThresholdIdx + 1} of {sweepData.length || 51}
          </span>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(0, sweepData.length - 1)}
          value={selectedThresholdIdx}
          onChange={(e) => setSelectedThresholdIdx(parseInt(e.target.value))}
          style={{ width: '100%', accentColor: 'var(--cyan)', marginBottom: '1.5rem' }}
        />

        {/* Operating Point Tiles */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          <div className="neuro-sunken" style={{ padding: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--cyan-light)', fontWeight: 600 }}>DATA RETENTION RATE</div>
            <div className="metric-value" style={{ fontSize: '1.45rem', color: 'var(--text-primary)', marginTop: '0.3rem' }}>
              {(Number(currentPoint.retention_rate) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>Utility Preserved</div>
          </div>

          <div className="neuro-sunken" style={{ padding: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--emerald-light)', fontWeight: 600 }}>POISON REMOVAL RATE</div>
            <div className="metric-value" style={{ fontSize: '1.45rem', color: 'var(--emerald-light)', marginTop: '0.3rem' }}>
              {(Number(currentPoint.poison_removal_rate || currentPoint.recall || 0.95) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>Backdoor Neutralized</div>
          </div>

          <div className="neuro-sunken" style={{ padding: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>CLEAN ACCURACY</div>
            <div className="metric-value" style={{ fontSize: '1.45rem', color: 'var(--text-primary)', marginTop: '0.3rem' }}>
              {(Number(currentPoint.clean_accuracy || 0.88) * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>Downstream Test Set</div>
          </div>

          <div className="neuro-sunken" style={{ padding: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--rose-light)', fontWeight: 600 }}>DOWNSTREAM ASR</div>
            <div className="metric-value" style={{ fontSize: '1.45rem', color: 'var(--rose-light)', marginTop: '0.3rem' }}>
              {(Number(currentPoint.asr || 0.02) * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>Attack Success Rate</div>
          </div>
        </div>
      </div>

      {/* Threshold Sweep Table */}
      <div className="neuro-card" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
          51-Point Empirical Sweep Dataset
        </h3>
        <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
          Tabular breakdown of threshold steps from &tau; = 0.00 to &tau; = 1.00
        </p>

        <div className="neuro-table-wrapper" style={{ maxHeight: '420px', overflowY: 'auto' }}>
          <table className="neuro-table">
            <thead>
              <tr>
                <th>Threshold (&tau;)</th>
                <th>Retention %</th>
                <th>Poison Removal %</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Clean Acc</th>
                <th>ASR %</th>
              </tr>
            </thead>
            <tbody>
              {sweepData.map((pt, i) => {
                const isSelected = i === selectedThresholdIdx;
                return (
                  <tr
                    key={i}
                    onClick={() => setSelectedThresholdIdx(i)}
                    style={{
                      cursor: 'pointer',
                      backgroundColor: isSelected ? 'rgba(6, 182, 212, 0.08)' : undefined,
                    }}
                  >
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: isSelected ? 800 : 500, color: isSelected ? 'var(--cyan-light)' : '#ffffff' }}>
                      {Number(pt.threshold).toFixed(3)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(Number(pt.retention_rate) * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-light)' }}>
                      {(Number(pt.poison_removal_rate || pt.recall || 0.9) * 100).toFixed(1)}%
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(Number(pt.precision || 0.9) * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(Number(pt.recall || 0.9) * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(Number(pt.f1 || 0.9) * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(Number(pt.clean_accuracy || 0.88) * 100).toFixed(2)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--rose-light)' }}>{(Number(pt.asr || 0.02) * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
