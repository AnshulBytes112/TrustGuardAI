import React, { useEffect, useState } from 'react';
import { fetchHealth, fetchOverviewStats, fetchSystemInfo, fetchResearchArtifacts } from '../../api';
import { LoadingSkeleton } from '../common/LoadingSkeleton';

export const SystemHealthView: React.FC = () => {
  const [healthData, setHealthData] = useState<{
    api: boolean;
    db: boolean;
    gpu: boolean;
    artifacts: boolean;
    latencyMs: number;
  }>({
    api: false,
    db: false,
    gpu: false,
    artifacts: false,
    latencyMs: 0,
  });
  const [loading, setLoading] = useState(true);

  const checkHealth = async () => {
    try {
      setLoading(true);
      const start = performance.now();

      const [health, stats, sys, arts] = await Promise.allSettled([
        fetchHealth(),
        fetchOverviewStats(),
        fetchSystemInfo(),
        fetchResearchArtifacts(),
      ]);

      const duration = Math.round(performance.now() - start);

      setHealthData({
        api: health.status === 'fulfilled' && health.value.status === 'ok',
        db: stats.status === 'fulfilled',
        gpu: sys.status === 'fulfilled',
        artifacts: arts.status === 'fulfilled',
        latencyMs: duration,
      });
    } catch (err) {
      console.error('Health check failed', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const nodes = [
    {
      name: 'FastAPI REST Server',
      desc: 'HTTP endpoint router, schema validation, asynchronous execution orchestration',
      status: healthData.api ? 'HEALTHY' : 'UNAVAILABLE',
      statusColor: healthData.api ? 'emerald' : 'rose',
    },
    {
      name: 'Relational Database (SQLite Engine)',
      desc: 'Dataset persistence, stratified sample indexing, and experiment telemetry history',
      status: healthData.db ? 'HEALTHY' : 'UNAVAILABLE',
      statusColor: healthData.db ? 'emerald' : 'rose',
    },
    {
      name: 'PyTorch ML Execution Engine',
      desc: 'DistilBERT representation extraction and neural classifier retraining pipeline',
      status: healthData.gpu ? 'HEALTHY' : 'DEGRADED',
      statusColor: healthData.gpu ? 'emerald' : 'amber',
    },
    {
      name: 'Server-Sent Events (SSE) Bus',
      desc: 'Real-time telemetry event streaming for live investigation console',
      status: healthData.api ? 'HEALTHY' : 'UNAVAILABLE',
      statusColor: healthData.api ? 'emerald' : 'rose',
    },
    {
      name: 'Artifacts Storage Subsystem',
      desc: 'Persistent research results, JSON sweep archives, and dataset artifacts',
      status: healthData.artifacts ? 'HEALTHY' : 'UNAVAILABLE',
      statusColor: healthData.artifacts ? 'emerald' : 'rose',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header Info */}
      <div className="neuro-card" style={{ padding: '1.5rem 1.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
            System Health & Node Status
          </h2>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', maxWidth: '820px', marginTop: '0.35rem' }}>
            Real-time diagnostics and connectivity state across all TrustGuardAI platform layers.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="neuro-badge cyan" style={{ fontFamily: 'var(--font-mono)' }}>
            Ping: {healthData.latencyMs} ms
          </div>
          <button onClick={checkHealth} className="neuro-btn neuro-btn-sm">
            Check Now
          </button>
        </div>
      </div>

      {/* Nodes Status List */}
      {loading ? (
        <LoadingSkeleton type="card" lines={5} />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {nodes.map((node) => (
            <div
              key={node.name}
              className="neuro-card"
              style={{
                padding: '1.25rem 1.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span className={`status-pip ${node.statusColor}`} />
                  <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
                    {node.name}
                  </h3>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                  {node.desc}
                </p>
              </div>

              <span className={`neuro-badge ${node.statusColor}`} style={{ fontSize: '0.76rem', padding: '0.3rem 0.75rem' }}>
                {node.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
