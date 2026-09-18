import React, { useEffect, useState } from 'react';
import { fetchLiveJobs } from '../../api';
import type { LiveJobSummary } from '../../api';
import { StatusBadge } from '../common/StatusBadge';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';
import { EmptyState } from '../common/EmptyState';

interface JobsHistoryViewProps {
  onInspectJob: (jobId: string) => void;
  onNewJob: () => void;
  searchQuery?: string;
}

export const JobsHistoryView: React.FC<JobsHistoryViewProps> = ({
  onInspectJob,
  onNewJob,
  searchQuery = '',
}) => {
  const [jobs, setJobs] = useState<LiveJobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('ALL');

  const loadJobs = async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await fetchLiveJobs();
      setJobs(list);
    } catch (err: any) {
      setError(err.message || 'Failed to load jobs history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadJobs();
  }, []);

  const filteredJobs = jobs.filter((j) => {
    if (statusFilter !== 'ALL' && j.status !== statusFilter) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        j.job_id.toLowerCase().includes(q) ||
        j.attack_type.toLowerCase().includes(q) ||
        j.dataset_id.toLowerCase().includes(q)
      );
    }
    return true;
  });

  if (loading && jobs.length === 0) {
    return <LoadingSkeleton type="table" lines={8} />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={loadJobs} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Action & Filter Bar */}
      <div className="neuro-card" style={{ padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {['ALL', 'RUNNING', 'COMPLETED', 'FAILED'].map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className="neuro-btn neuro-btn-sm"
              style={{
                backgroundColor: statusFilter === st ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
                borderColor: statusFilter === st ? 'var(--cyan-border)' : 'var(--border-subtle)',
                color: statusFilter === st ? 'var(--cyan-light)' : 'var(--text-muted)',
              }}
            >
              {st}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={loadJobs} className="neuro-btn neuro-btn-sm">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
              <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
              <path d="M8 16H3v5" />
            </svg>
            <span>Refresh</span>
          </button>
          <button onClick={onNewJob} className="neuro-btn neuro-btn-primary neuro-btn-sm">
            <span>+ New Investigation</span>
          </button>
        </div>
      </div>

      {/* Jobs Table */}
      {filteredJobs.length === 0 ? (
        <EmptyState
          title="No Investigation Jobs Found"
          description="There are no jobs matching the selected criteria. Launch a new investigation to start scanning."
          actionText="Launch New Investigation"
          onAction={onNewJob}
        />
      ) : (
        <div className="neuro-table-wrapper">
          <table className="neuro-table">
            <thead>
              <tr>
                <th>Job Identifier</th>
                <th>Dataset</th>
                <th>Attack Mechanism</th>
                <th>Poison Rate</th>
                <th>Current Stage</th>
                <th>Status</th>
                <th>Started</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((j) => (
                <tr key={j.job_id}>
                  <td>
                    <span className="code-badge">{j.job_id.substring(0, 8)}...</span>
                  </td>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    {j.dataset_id}
                  </td>
                  <td style={{ textTransform: 'capitalize', color: 'var(--text-secondary)' }}>
                    {j.attack_type.replace(/_/g, ' ')}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {(j.poison_rate * 100).toFixed(1)}%
                  </td>
                  <td style={{ color: 'var(--cyan-light)', fontSize: '0.78rem' }}>
                    {j.current_stage || 'N/A'}
                  </td>
                  <td>
                    <StatusBadge status={j.status} />
                  </td>
                  <td style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                    {new Date(j.created_at).toLocaleTimeString()}
                  </td>
                  <td>
                    <button
                      onClick={() => onInspectJob(j.job_id)}
                      className="neuro-btn neuro-btn-sm"
                      style={{ fontSize: '0.74rem', padding: '0.25rem 0.65rem' }}
                    >
                      {j.status === 'RUNNING' ? 'Watch Live' : 'View Results'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
