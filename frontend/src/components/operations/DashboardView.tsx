import React, { useEffect, useState } from 'react';
import { fetchOverviewStats, fetchLiveJobs } from '../../api';
import type { OverviewStats, LiveJobSummary, SystemInfo } from '../../api';
import { MetricCard } from '../common/MetricCard';
import { StatusBadge } from '../common/StatusBadge';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';

interface DashboardViewProps {
  onNavigate: (tabId: string, jobId?: string) => void;
  systemInfo: SystemInfo | null;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate, systemInfo }) => {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [jobs, setJobs] = useState<LiveJobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [statsData, jobsData] = await Promise.all([
        fetchOverviewStats(),
        fetchLiveJobs(),
      ]);
      setStats(statsData);
      setJobs(jobsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load command center data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton type="card" height="120px" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
          <LoadingSkeleton type="card" height="140px" />
          <LoadingSkeleton type="card" height="140px" />
          <LoadingSkeleton type="card" height="140px" />
          <LoadingSkeleton type="card" height="140px" />
        </div>
      </div>
    );
  }

  if (error) {
    return <ErrorState message={error} onRetry={loadDashboardData} />;
  }

  const activeJobs = jobs.filter((j) => j.status === 'RUNNING' || j.status === 'CREATED');
  const completedJobs = jobs.filter((j) => j.status === 'COMPLETED');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Hero Security Overview Card */}
      <div
        className="neuro-card"
        style={{
          padding: '1.6rem 2rem',
          background: 'linear-gradient(135deg, rgba(15, 22, 38, 0.95) 0%, rgba(21, 29, 50, 0.95) 100%)',
          border: '1px solid var(--border-elevated)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1.5rem',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
            <span className="status-pip emerald" />
            <span style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--emerald-light)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              Training-Data Integrity Guard Active
            </span>
          </div>
          <h2
            style={{
              fontFamily: 'var(--font-brand)',
              fontSize: '1.5rem',
              fontWeight: 800,
              color: '#ffffff',
              letterSpacing: '-0.02em',
            }}
          >
            AI Security Command Center
          </h2>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', maxWidth: '640px', marginTop: '0.25rem' }}>
            Continuous backdoor poison detection, multi-signal representation consistency analysis, and zero-leakage dataset purification.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            onClick={() => onNavigate('investigate_new')}
            className="neuro-btn neuro-btn-primary"
            style={{ fontWeight: 700 }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            <span>Launch Investigation</span>
          </button>
          <button
            onClick={() => onNavigate('datasets')}
            className="neuro-btn"
          >
            <span>Upload Dataset</span>
          </button>
        </div>
      </div>

      {/* KPI Metrics Row */}
      <div className="dashboard-grid">
        <MetricCard
          title="TOTAL DATASETS"
          value={stats?.total_datasets || 0}
          subtitle="Ingested in database"
          badgeText="Active"
          badgeTone="cyan"
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <ellipse cx="12" cy="5" rx="9" ry="3" />
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
            </svg>
          }
        />

        <MetricCard
          title="PROCESSED SAMPLES"
          value={stats?.total_samples.toLocaleString() || '0'}
          subtitle={`${stats?.total_quarantined || 0} quarantined`}
          badgeText={stats?.total_quarantined ? 'Anomalies Isolated' : 'Clean'}
          badgeTone={stats?.total_quarantined ? 'amber' : 'emerald'}
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          }
        />

        <MetricCard
          title="INVESTIGATION JOBS"
          value={jobs.length}
          subtitle={`${activeJobs.length} active, ${completedJobs.length} completed`}
          badgeText={activeJobs.length > 0 ? `${activeJobs.length} Running` : 'Idle'}
          badgeTone={activeJobs.length > 0 ? 'cyan' : 'emerald'}
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          }
        />

        <MetricCard
          title="COMPUTE ENGINE"
          value={systemInfo?.gpu_available ? 'CUDA GPU' : 'CPU Engine'}
          subtitle={systemInfo?.gpu_available ? `${systemInfo.vram_total_gb} GB VRAM` : 'PyTorch Multi-Thread'}
          badgeText={systemInfo?.gpu_available ? 'Accelerated' : 'Fallback'}
          badgeTone={systemInfo?.gpu_available ? 'violet' : 'cyan'}
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="16" height="16" x="4" y="4" rx="2" />
              <rect width="6" height="6" x="9" y="9" rx="1" />
            </svg>
          }
        />
      </div>

      {/* Two Column Grid: Active & Recent Live Jobs + Dataset Inventory */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem' }}>
        {/* Recent Live Investigations */}
        <div className="neuro-card" style={{ padding: '1.4rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.2rem' }}>
            <div>
              <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
                Recent Investigation Pipelines
              </h3>
              <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Real-time multi-signal defense runs and SSE telemetry sessions
              </p>
            </div>
            <button onClick={() => onNavigate('jobs')} className="neuro-btn neuro-btn-sm">
              View All Jobs
            </button>
          </div>

          {jobs.length === 0 ? (
            <div className="neuro-sunken" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              No investigation jobs registered yet. Launch your first pipeline to begin.
            </div>
          ) : (
            <div className="neuro-table-wrapper">
              <table className="neuro-table">
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Attack Type</th>
                    <th>Poison Rate</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.slice(0, 5).map((job) => (
                    <tr key={job.job_id}>
                      <td>
                        <span className="code-badge">{job.job_id.substring(0, 8)}...</span>
                      </td>
                      <td style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--text-secondary)' }}>
                        {job.attack_type.replace(/_/g, ' ')}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>
                        {(job.poison_rate * 100).toFixed(1)}%
                      </td>
                      <td>
                        <StatusBadge status={job.status} />
                      </td>
                      <td>
                        <button
                          onClick={() => onNavigate('live_pipeline', job.job_id)}
                          className="neuro-btn neuro-btn-sm"
                          style={{ fontSize: '0.74rem', padding: '0.25rem 0.6rem' }}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Dataset Inventory Overview */}
        <div className="neuro-card" style={{ padding: '1.4rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.2rem' }}>
            <div>
              <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
                Dataset Inventory
              </h3>
              <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Ingested training corpora and split distributions
              </p>
            </div>
            <button onClick={() => onNavigate('datasets')} className="neuro-btn neuro-btn-sm">
              Manage
            </button>
          </div>

          {stats?.recent_datasets && stats.recent_datasets.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {stats.recent_datasets.slice(0, 4).map((d) => (
                <div
                  key={d.id}
                  className="neuro-sunken"
                  style={{
                    padding: '0.85rem 1rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.86rem', color: 'var(--text-primary)' }}>
                      {d.name}
                    </div>
                    <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                      {d.total_samples} samples &bull; {d.train_count} train / {d.test_count} test
                    </div>
                  </div>
                  <span className="neuro-badge cyan" style={{ fontSize: '0.68rem' }}>
                    {d.modality}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="neuro-sunken" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              No datasets uploaded. Ingest SST-2 or custom data to start.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
