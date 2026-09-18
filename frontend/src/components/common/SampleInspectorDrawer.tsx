import React, { useEffect } from 'react';
import type { LiveSampleInspection } from '../../api';
import { StatusBadge } from './StatusBadge';

interface SampleInspectorDrawerProps {
  sample: LiveSampleInspection | null;
  onClose: () => void;
}

export const SampleInspectorDrawer: React.FC<SampleInspectorDrawerProps> = ({
  sample,
  onClose,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    if (sample) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sample, onClose]);

  if (!sample) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          backdropFilter: 'blur(4px)',
          zIndex: 110,
        }}
        aria-hidden="true"
      />

      {/* Slide-in Panel */}
      <div
        className="neuro-card"
        role="dialog"
        aria-modal="true"
        aria-label={`Sample Explainability & Attribution for ${sample.sample_id}`}
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          width: '580px',
          maxWidth: '92vw',
          height: '100vh',
          backgroundColor: 'var(--bg-surface-elevated)',
          borderLeft: '1px solid var(--border-elevated)',
          zIndex: 120,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--shadow-elevated)',
          overflow: 'hidden',
          animation: 'fadeIn 0.2s ease-out',
        }}
      >
        {/* Drawer Header */}
        <div
          style={{
            padding: '1.4rem 1.6rem',
            borderBottom: '1px solid var(--border-default)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: 'var(--bg-surface)',
          }}
        >
          <div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
              Sample Explainability & Attribution
            </div>
            <div
              style={{
                fontFamily: 'var(--font-brand)',
                fontSize: '1.15rem',
                fontWeight: 700,
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
              }}
            >
              <span>Sample {sample.sample_id}</span>
              <StatusBadge status={sample.decision} size="sm" />
            </div>
          </div>
          <button
            onClick={onClose}
            className="neuro-btn neuro-btn-sm"
            style={{ padding: '0.4rem', borderRadius: 'var(--radius-sm)' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Drawer Body */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1.6rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
          }}
        >
          {/* Sample Text Surface */}
          <div className="neuro-card" style={{ padding: '1.2rem', backgroundColor: 'var(--bg-sunken)' }}>
            <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              RAW TRAINING TEXT
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.92rem',
                lineHeight: 1.6,
                color: 'var(--text-primary)',
              }}
            >
              {sample.text}
            </div>
            <div
              style={{
                marginTop: '0.9rem',
                paddingTop: '0.75rem',
                borderTop: '1px solid var(--border-subtle)',
                display: 'flex',
                gap: '1rem',
                fontSize: '0.76rem',
                color: 'var(--text-dim)',
              }}
            >
              <div>
                Label: <strong style={{ color: 'var(--text-secondary)' }}>{sample.label || 'None'}</strong>
              </div>
              <div>
                Split: <strong style={{ color: 'var(--text-secondary)' }}>{sample.split}</strong>
              </div>
              {sample.ground_truth_poisoned !== undefined && sample.ground_truth_poisoned !== null && (
                <div>
                  <span style={{ color: 'var(--text-dim)' }}>Benchmark Ground Truth: </span>
                  <strong style={{ color: sample.ground_truth_poisoned ? 'var(--rose)' : 'var(--emerald)' }}>
                    {sample.ground_truth_poisoned ? 'POISONED' : 'CLEAN'}
                  </strong>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontStyle: 'italic', marginTop: '0.15rem' }}>
                    (Benchmark oracle label — inaccessible during defensive inference)
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Scores Overview */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="neuro-sunken" style={{ padding: '1rem 1.2rem' }}>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontWeight: 600 }}>Suspicion Score (S_i)</div>
              <div
                className="metric-value"
                style={{
                  fontSize: '1.6rem',
                  fontWeight: 800,
                  color: sample.suspicion_score > sample.threshold ? 'var(--rose)' : 'var(--emerald)',
                  marginTop: '0.2rem',
                }}
              >
                {sample.suspicion_score.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                Threshold: {sample.threshold.toFixed(4)}
              </div>
            </div>

            <div className="neuro-sunken" style={{ padding: '1rem 1.2rem' }}>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontWeight: 600 }}>Trust Score (T_i)</div>
              <div
                className="metric-value"
                style={{
                  fontSize: '1.6rem',
                  fontWeight: 800,
                  color: 'var(--cyan-light)',
                  marginTop: '0.2rem',
                }}
              >
                {sample.trust_score.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                T_i = 1 - S_i
              </div>
            </div>
          </div>

          {/* Signal Decomposition */}
          <div className="neuro-card" style={{ padding: '1.25rem' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1rem' }}>
              Four-Signal Decomposition & Linear Contribution
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              {sample.contributions && sample.contributions.length > 0 ? (
                sample.contributions.map((c) => {
                  const pct = Math.max(0, Math.min(100, Math.round(c.contribution_percentage)));
                  return (
                    <div key={c.signal_name}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.3rem' }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                          {c.signal_name} (w = {c.weight.toFixed(2)})
                        </span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                          {c.normalized_value.toFixed(3)} ({pct}%)
                        </span>
                      </div>
                      <div
                        style={{
                          height: '6px',
                          backgroundColor: 'var(--bg-sunken)',
                          borderRadius: 'var(--radius-full)',
                          overflow: 'hidden',
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        <div
                          style={{
                            height: '100%',
                            width: `${pct}%`,
                            backgroundColor: 'var(--cyan)',
                            borderRadius: 'var(--radius-full)',
                          }}
                        />
                      </div>
                    </div>
                  );
                })
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                  Signal attribution weights available:
                  <ul style={{ paddingLeft: '1.2rem', marginTop: '0.5rem' }}>
                    <li>Semantic: {sample.signals.semantic !== undefined ? sample.signals.semantic?.toFixed(3) : 'N/A'}</li>
                    <li>Neighborhood: {sample.signals.neighborhood !== undefined ? sample.signals.neighborhood?.toFixed(3) : 'N/A'}</li>
                    <li>Stability: {sample.signals.stability !== undefined ? sample.signals.stability?.toFixed(3) : 'N/A'}</li>
                    <li>Density: {sample.signals.density !== undefined ? sample.signals.density?.toFixed(3) : 'N/A'}</li>
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* Research Invariant Explanation */}
          <div className="neuro-sunken" style={{ padding: '1rem 1.2rem', fontSize: '0.78rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>
            <strong style={{ color: 'var(--text-secondary)' }}>Security Invariant:</strong> If suspicion score exceeds calibrated threshold ({sample.threshold.toFixed(4)}), the sample is automatically quarantined into isolated storage and excluded from downstream model retraining.
          </div>
        </div>
      </div>
    </>
  );
};
