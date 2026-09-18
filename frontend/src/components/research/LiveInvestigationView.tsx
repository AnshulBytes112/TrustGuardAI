import React, { useEffect, useState, useRef } from 'react';
import {
  fetchLiveJob,
  createLiveEventSource,
} from '../../api';
import type {
  LiveJobResponse,
  LiveSSEEvent,
  LiveSampleInspection,
} from '../../api';
import { StatusBadge } from '../common/StatusBadge';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';
import { SampleInspectorDrawer } from '../common/SampleInspectorDrawer';

interface LiveInvestigationViewProps {
  jobId: string;
  onBackToJobs: () => void;
  onNewJob: () => void;
}

const PIPELINE_STAGES = [
  { id: 'DATASET', label: 'Dataset Ingestion', icon: '📁' },
  { id: 'VALIDATION', label: 'Schema Validation', icon: '🛡️' },
  { id: 'POISONING', label: 'Attack Injection', icon: '💉' },
  { id: 'SPLITTING', label: 'Stratified Split', icon: '✂️' },
  { id: 'REPRESENTATIONS', label: 'DistilBERT Embeddings', icon: '🧠' },
  { id: 'SEMANTIC', label: 'Semantic Consistency', icon: '📐' },
  { id: 'NEIGHBORHOOD', label: 'Neighborhood Purity', icon: '👥' },
  { id: 'STABILITY', label: 'Prediction Stability', icon: '⚖️' },
  { id: 'DENSITY', label: 'Manifold Density', icon: '📊' },
  { id: 'CALIBRATION', label: 'Threshold Calibration', icon: '🎯' },
  { id: 'ISOLATION', label: 'Quarantine Isolation', icon: '🔒' },
  { id: 'RETRAINING', label: 'Downstream Retraining', icon: '⚡' },
  { id: 'EVALUATION', label: 'Security Evaluation', icon: '📈' },
  { id: 'BASELINES', label: 'FLARE & ONION Baselines', icon: '🔬' },
];

export const LiveInvestigationView: React.FC<LiveInvestigationViewProps> = ({
  jobId,
  onBackToJobs,
  onNewJob,
}) => {
  const [job, setJob] = useState<LiveJobResponse | null>(null);
  const [events, setEvents] = useState<LiveSSEEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Telemetry Log Controls
  const [autoScroll, setAutoScroll] = useState(true);
  const [logFilter, setLogFilter] = useState('');
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Sample Table & Inspection State
  const [selectedSample, setSelectedSample] = useState<LiveSampleInspection | null>(null);
  const [sampleDecisionFilter, setSampleDecisionFilter] = useState<'ALL' | 'ISOLATE' | 'RETAIN'>('ALL');
  const [sampleSearch, setSampleSearch] = useState('');
  const [samplePage, setSamplePage] = useState(1);
  const pageSize = 12;

  // Initial Job Load and SSE subscription
  useEffect(() => {
    let es: EventSource | null = null;
    let pollInterval: any = null;

    const init = async () => {
      try {
        setLoading(true);
        setError(null);
        const jobData = await fetchLiveJob(jobId);
        setJob(jobData);
        setEvents(jobData.events || []);

        if (jobData.status === 'RUNNING' || jobData.status === 'CREATED') {
          // 1. Connect real-time SSE stream
          es = createLiveEventSource(
            jobId,
            (newEvent) => {
              setEvents((prev) => {
                if (prev.some((e) => e.event_id === newEvent.event_id && e.event_type === newEvent.event_type)) {
                  return prev;
                }
                return [...prev, newEvent];
              });

              if (newEvent.event_type === 'JOB_COMPLETED' || newEvent.event_type === 'JOB_FAILED' || newEvent.event_type.endsWith('_COMPLETED')) {
                fetchLiveJob(jobId).then((updated) => {
                  setJob(updated);
                  if (updated.status === 'COMPLETED' || updated.status === 'FAILED') {
                    if (es) es.close();
                    if (pollInterval) clearInterval(pollInterval);
                  }
                }).catch(console.error);
              }
            },
            (err) => {
              console.warn('SSE EventSource dropped/interrupted; background polling active', err);
            }
          );

          // 2. Resilient fallback polling every 3.5s while running to guarantee terminal state capture
          pollInterval = setInterval(async () => {
            try {
              const updated = await fetchLiveJob(jobId);
              setJob(updated);
              setEvents(updated.events || []);
              if (updated.status === 'COMPLETED' || updated.status === 'FAILED') {
                clearInterval(pollInterval);
                if (es) es.close();
              }
            } catch (err) {
              console.error('Job polling error:', err);
            }
          }, 3500);
        }
      } catch (err: any) {
        setError(err.message || `Failed to load live investigation ${jobId}`);
      } finally {
        setLoading(false);
      }
    };

    init();

    return () => {
      if (es) es.close();
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [jobId]);

  // Auto scroll telemetry logs
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  if (loading && !job) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton type="card" height="180px" />
        <LoadingSkeleton type="card" height="300px" />
      </div>
    );
  }

  if (error && !job) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  }

  const isCompleted = job?.status === 'COMPLETED';
  const isRunning = job?.status === 'RUNNING' || job?.status === 'CREATED';

  // Determine stage status
  const currentStageName = job?.current_stage || 'INITIALIZING';

  const filteredEvents = events.filter((e) => {
    if (!logFilter) return true;
    const q = logFilter.toLowerCase();
    return (
      e.event_type.toLowerCase().includes(q) ||
      e.message.toLowerCase().includes(q) ||
      e.stage.toLowerCase().includes(q)
    );
  });

  const sampleInspections = job?.sample_inspections || [];
  const filteredSamples = sampleInspections.filter((s) => {
    if (sampleDecisionFilter !== 'ALL' && s.decision !== sampleDecisionFilter) return false;
    if (sampleSearch && !s.text.toLowerCase().includes(sampleSearch.toLowerCase()) && !s.sample_id.includes(sampleSearch)) {
      return false;
    }
    return true;
  });

  const totalSamplePages = Math.ceil(filteredSamples.length / pageSize) || 1;
  const paginatedSamples = filteredSamples.slice((samplePage - 1) * pageSize, samplePage * pageSize);

  const retraining = job?.retraining_report;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Sample Drawer Modal */}
      <SampleInspectorDrawer sample={selectedSample} onClose={() => setSelectedSample(null)} />

      {/* Header Bar */}
      <div
        className="neuro-card"
        style={{
          padding: '1.5rem 1.75rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1.25rem',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
            <span className="code-badge" style={{ fontSize: '0.8rem' }}>JOB #{job?.job_id.substring(0, 10)}</span>
            <StatusBadge status={job?.status || 'PENDING'} size="md" />
            <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
              Dataset: <strong style={{ color: 'var(--text-secondary)' }}>{job?.request?.dataset_id}</strong> &bull; Attack: <strong style={{ color: 'var(--rose-light)' }}>{job?.request?.attack_type}</strong> ({( (job?.request?.poison_rate || 0) * 100 ).toFixed(1)}%)
            </span>
          </div>
          <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
            Live Multi-Signal Investigation Pipeline
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={onBackToJobs} className="neuro-btn neuro-btn-sm">
            ← Jobs History
          </button>
          <button onClick={onNewJob} className="neuro-btn neuro-btn-primary neuro-btn-sm">
            + New Run
          </button>
        </div>
      </div>

      {/* Pipeline Stage Topology Flow */}
      <div className="neuro-card" style={{ padding: '1.5rem' }}>
        <div style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-dim)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '1.2rem' }}>
          Real-Time Pipeline Execution Topology
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '0.75rem' }}>
          {PIPELINE_STAGES.map((stage) => {
            const hasCompleted = isCompleted || events.some((e) => e.stage.toUpperCase().includes(stage.id) && e.status === 'COMPLETED');
            const isStageActive = isRunning && currentStageName.toUpperCase().includes(stage.id);

            return (
              <div
                key={stage.id}
                className={hasCompleted || isStageActive ? 'neuro-card' : 'neuro-sunken'}
                style={{
                  padding: '0.85rem 0.65rem',
                  textAlign: 'center',
                  border: isStageActive
                    ? '1px solid var(--cyan)'
                    : hasCompleted
                    ? '1px solid var(--emerald-border)'
                    : '1px solid var(--border-subtle)',
                  backgroundColor: isStageActive
                    ? 'var(--cyan-dim)'
                    : hasCompleted
                    ? 'rgba(16, 185, 129, 0.08)'
                    : 'var(--bg-sunken)',
                  boxShadow: isStageActive ? 'var(--glow-cyan)' : 'none',
                  transition: 'all var(--transition-smooth)',
                }}
              >
                <div style={{ fontSize: '1.2rem', marginBottom: '0.3rem' }}>{stage.icon}</div>
                <div style={{ fontSize: '0.74rem', fontWeight: 700, color: isStageActive ? '#ffffff' : hasCompleted ? 'var(--emerald-light)' : 'var(--text-muted)' }}>
                  {stage.label}
                </div>
                <div style={{ marginTop: '0.35rem' }}>
                  {isStageActive ? (
                    <span className="neuro-badge cyan" style={{ fontSize: '0.62rem', padding: '0.1rem 0.35rem' }}>
                      <span className="status-pip cyan" /> Active
                    </span>
                  ) : hasCompleted ? (
                    <span className="neuro-badge emerald" style={{ fontSize: '0.62rem', padding: '0.1rem 0.35rem' }}>
                      Done
                    </span>
                  ) : (
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>Queued</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Real-time Telemetry Log Terminal */}
      <div className="neuro-card" style={{ padding: '1.4rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.9rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span className={`status-pip ${isRunning ? 'cyan' : 'emerald'}`} />
            <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
              Real-Time Server-Sent Events Telemetry ({events.length} events)
            </h3>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <input
              type="text"
              placeholder="Filter telemetry log..."
              value={logFilter}
              onChange={(e) => setLogFilter(e.target.value)}
              className="neuro-input"
              style={{ fontSize: '0.76rem', padding: '0.3rem 0.6rem', width: '180px' }}
            />

            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className="neuro-btn neuro-btn-sm"
              style={{
                fontSize: '0.72rem',
                backgroundColor: autoScroll ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
                color: autoScroll ? 'var(--cyan-light)' : 'var(--text-muted)',
              }}
            >
              {autoScroll ? 'Auto-Scroll: ON' : 'Auto-Scroll: OFF'}
            </button>

            <button
              onClick={() => {
                const text = events.map((e) => `[${new Date(e.timestamp).toLocaleTimeString()}] ${e.stage} - ${e.message}`).join('\n');
                navigator.clipboard.writeText(text);
              }}
              className="neuro-btn neuro-btn-sm"
              style={{ fontSize: '0.72rem' }}
            >
              Copy Logs
            </button>
          </div>
        </div>

        <div ref={logContainerRef} className="telemetry-terminal">
          {filteredEvents.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '2rem 0' }}>
              Awaiting telemetry streaming from worker pipeline...
            </div>
          ) : (
            filteredEvents.map((ev, idx) => (
              <div key={ev.event_id || idx} className="telemetry-row">
                <span className="telemetry-time">[{new Date(ev.timestamp).toLocaleTimeString()}]</span>
                <span className="telemetry-stage">[{ev.stage}]</span>
                <span style={{ color: ev.event_type.includes('COMPLETED') ? 'var(--emerald-light)' : ev.event_type.includes('FAILED') ? 'var(--rose-light)' : 'var(--text-secondary)' }}>
                  {ev.message}
                </span>
                {ev.progress_info && ev.progress_info.processed_samples !== undefined && (
                  <span style={{ color: 'var(--cyan-light)', marginLeft: 'auto', flexShrink: 0 }}>
                    ({ev.progress_info.processed_samples}/{ev.progress_info.total_samples})
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Live Security Metrics & Retraining Evaluation */}
      {retraining && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          {/* Model A vs Model B Side-by-Side */}
          <div className="neuro-card" style={{ padding: '1.5rem' }}>
            <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
              Downstream Model Retraining Benchmark
            </h3>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
              Model A (Contaminated Raw Training) vs Model B (TrustGuard Purified Training)
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
              {/* Model A */}
              <div className="neuro-sunken" style={{ padding: '1rem', border: '1px solid var(--rose-border)' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--rose-light)', textTransform: 'uppercase' }}>
                  Model A (Contaminated)
                </div>
                <div style={{ marginTop: '0.6rem' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Clean Accuracy</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--text-primary)' }}>
                    {(retraining.baseline_clean_accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div style={{ marginTop: '0.6rem' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--rose-light)' }}>Attack Success Rate (ASR)</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--rose-light)' }}>
                    {(retraining.baseline_attack_success_rate * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Model B */}
              <div className="neuro-sunken" style={{ padding: '1rem', border: '1px solid var(--emerald-border)', backgroundColor: 'rgba(16, 185, 129, 0.04)' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--emerald-light)', textTransform: 'uppercase' }}>
                  Model B (TrustGuard Purified)
                </div>
                <div style={{ marginTop: '0.6rem' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Clean Accuracy</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--emerald-light)' }}>
                    {(retraining.purified_clean_accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div style={{ marginTop: '0.6rem' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--emerald-light)' }}>Attack Success Rate (ASR)</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--emerald-light)' }}>
                    {(retraining.purified_attack_success_rate * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Metric Deltas */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
              <div className="neuro-sunken" style={{ padding: '0.75rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>ASR REDUCTION</div>
                <div className="metric-value" style={{ fontSize: '1.25rem', color: 'var(--emerald-light)', marginTop: '0.2rem' }}>
                  {(retraining.attack_success_rate_reduction * 100).toFixed(2)}%
                </div>
              </div>

              <div className="neuro-sunken" style={{ padding: '0.75rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>ACCURACY DELTA</div>
                <div className="metric-value" style={{ fontSize: '1.25rem', color: retraining.clean_accuracy_delta >= 0 ? 'var(--emerald-light)' : 'var(--amber-light)', marginTop: '0.2rem' }}>
                  {retraining.clean_accuracy_delta >= 0 ? '+' : ''}{(retraining.clean_accuracy_delta * 100).toFixed(2)}%
                </div>
              </div>

              <div className="neuro-sunken" style={{ padding: '0.75rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>RETENTION RATE</div>
                <div className="metric-value" style={{ fontSize: '1.25rem', color: 'var(--cyan-light)', marginTop: '0.2rem' }}>
                  {(retraining.retention_rate * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Defense Calibration & Learned Weights */}
          <div className="neuro-card" style={{ padding: '1.5rem' }}>
            <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
              TrustGuard Calibration & Signal Weights
            </h3>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginBottom: '1.25rem' }}>
              Optimal threshold calibrated on validation split without test set contamination
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              <div className="neuro-sunken" style={{ padding: '0.9rem 1.1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>CALIBRATED THRESHOLD (&tau;)</div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--cyan-light)', fontFamily: 'var(--font-mono)' }}>
                    {job.calibrated_threshold !== undefined && job.calibrated_threshold !== null ? job.calibrated_threshold.toFixed(4) : 'N/A'}
                  </div>
                </div>
                <span className="neuro-badge cyan">
                  {job.request?.calibration_method || 'Youden J'}
                </span>
              </div>

              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                Learned Signal Contributions:
              </div>

              {job.learned_weights ? (
                Object.entries(job.learned_weights).map(([signal, weight]) => (
                  <div key={signal}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: '0.25rem' }}>
                      <span style={{ textTransform: 'capitalize', color: 'var(--text-secondary)', fontWeight: 600 }}>
                        {signal} Signal
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        w = {Number(weight).toFixed(3)}
                      </span>
                    </div>
                    <div style={{ height: '5px', backgroundColor: 'var(--bg-sunken)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(100, Math.round(Number(weight) * 100))}%`,
                          backgroundColor: 'var(--violet)',
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                  Weights calibrated uniformly across 4 signals (0.25 each).
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Research Baselines Comparison Table */}
      {job?.baseline_results && job.baseline_results.length > 0 && (
        <div className="neuro-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
            Empirical Baseline Comparison (Neutral Presentation)
          </h3>
          <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginBottom: '1rem' }}>
            Comparative evaluation against published backdoor defense mechanisms on identical splits
          </p>

          <div className="neuro-table-wrapper">
            <table className="neuro-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>AUROC</th>
                  <th>Retention</th>
                  <th>Downstream Clean Acc</th>
                  <th>Downstream ASR</th>
                  <th>Runtime (s)</th>
                </tr>
              </thead>
              <tbody>
                {job.baseline_results.map((b) => (
                  <tr key={b.method}>
                    <td style={{ fontWeight: 700, color: '#ffffff' }}>
                      {b.method.toUpperCase()}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(b.precision * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{b.recall !== null && b.recall !== undefined ? `${(b.recall * 100).toFixed(1)}%` : 'N/A'}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(b.f1 * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{b.auroc !== null && b.auroc !== undefined ? (b.auroc).toFixed(3) : 'N/A'}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(b.retention_rate * 100).toFixed(1)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{(b.downstream_clean_accuracy * 100).toFixed(2)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--rose-light)' }}>{(b.downstream_attack_success_rate * 100).toFixed(2)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{b.runtime_seconds.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Sample Attribution & Quarantined Items Explorer */}
      {sampleInspections.length > 0 && (
        <div className="neuro-card" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>
                Sample-Level Security Inspections ({sampleInspections.length} Total)
              </h3>
              <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Click any row to inspect deep multi-signal XAI attribution and linear contribution breakdown
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              {(['ALL', 'ISOLATE', 'RETAIN'] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => {
                    setSampleDecisionFilter(d);
                    setSamplePage(1);
                  }}
                  className="neuro-btn neuro-btn-sm"
                  style={{
                    backgroundColor: sampleDecisionFilter === d ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
                    borderColor: sampleDecisionFilter === d ? 'var(--cyan-border)' : 'var(--border-subtle)',
                    color: sampleDecisionFilter === d ? 'var(--cyan-light)' : 'var(--text-muted)',
                  }}
                >
                  {d === 'ALL' ? 'All Samples' : d === 'ISOLATE' ? 'Quarantined Poison' : 'Retained Clean'}
                </button>
              ))}

              <input
                type="text"
                placeholder="Search sample text..."
                value={sampleSearch}
                onChange={(e) => {
                  setSampleSearch(e.target.value);
                  setSamplePage(1);
                }}
                className="neuro-input"
                style={{ fontSize: '0.76rem', padding: '0.3rem 0.6rem', width: '190px' }}
              />
            </div>
          </div>

          <div className="neuro-table-wrapper">
            <table className="neuro-table">
              <thead>
                <tr>
                  <th style={{ width: '90px' }}>ID</th>
                  <th>Training Text Snippet</th>
                  <th style={{ width: '90px' }}>Label</th>
                  <th style={{ width: '130px' }}>Benchmark Truth (Oracle)</th>
                  <th style={{ width: '110px' }}>Suspicion S_i</th>
                  <th style={{ width: '100px' }}>Trust T_i</th>
                  <th style={{ width: '120px' }}>Defense Decision</th>
                  <th style={{ width: '90px' }}>XAI Action</th>
                </tr>
              </thead>
              <tbody>
                {paginatedSamples.map((s) => (
                  <tr
                    key={s.sample_id}
                    onClick={() => setSelectedSample(s)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <span className="code-badge">{s.sample_id}</span>
                    </td>
                    <td style={{ maxWidth: '340px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={s.text}>
                      {s.text}
                    </td>
                    <td>
                      <span className="code-badge">{s.label || 'N/A'}</span>
                    </td>
                    <td>
                      {s.ground_truth_poisoned !== undefined && s.ground_truth_poisoned !== null ? (
                        <span
                          className={`neuro-badge ${s.ground_truth_poisoned ? 'rose' : 'emerald'}`}
                          style={{ fontSize: '0.65rem' }}
                          title="Benchmark Ground Truth Label (Evaluation oracle only)"
                        >
                          {s.ground_truth_poisoned ? 'POISON' : 'CLEAN'}
                        </span>
                      ) : (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>N/A</span>
                      )}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: s.suspicion_score > s.threshold ? 'var(--rose-light)' : 'var(--text-secondary)' }}>
                      {s.suspicion_score.toFixed(4)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan-light)' }}>
                      {s.trust_score.toFixed(4)}
                    </td>
                    <td>
                      <StatusBadge status={s.decision} size="sm" />
                    </td>
                    <td>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedSample(s);
                        }}
                        className="neuro-btn neuro-btn-sm"
                        style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Sample Pagination */}
          {totalSamplePages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.75rem' }}>
              <button
                disabled={samplePage === 1}
                onClick={() => setSamplePage((p) => Math.max(1, p - 1))}
                className="neuro-btn neuro-btn-sm"
              >
                Previous
              </button>
              <span style={{ display: 'flex', alignItems: 'center', fontSize: '0.76rem', color: 'var(--text-dim)', padding: '0 0.5rem' }}>
                Page {samplePage} of {totalSamplePages}
              </span>
              <button
                disabled={samplePage === totalSamplePages}
                onClick={() => setSamplePage((p) => Math.min(totalSamplePages, p + 1))}
                className="neuro-btn neuro-btn-sm"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
