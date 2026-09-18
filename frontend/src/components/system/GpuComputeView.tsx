import React, { useEffect, useState } from 'react';
import { fetchSystemInfo } from '../../api';
import type { SystemInfo } from '../../api';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';

interface GpuComputeViewProps {
  systemInfo: SystemInfo | null;
}

export const GpuComputeView: React.FC<GpuComputeViewProps> = ({ systemInfo: initialInfo }) => {
  const [info, setInfo] = useState<SystemInfo | null>(initialInfo);
  const [loading, setLoading] = useState(!initialInfo);
  const [error, setError] = useState<string | null>(null);

  const loadInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchSystemInfo();
      setInfo(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch compute telemetry');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!initialInfo) {
      loadInfo();
    }
  }, [initialInfo]);

  if (loading && !info) {
    return <LoadingSkeleton type="card" height="300px" />;
  }

  if (error && !info) {
    return <ErrorState message={error} onRetry={loadInfo} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header Info */}
      <div className="neuro-card" style={{ padding: '1.5rem 1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
              <span className={`status-pip ${info?.gpu_available ? 'cyan' : 'dim'}`} />
              <span style={{ fontSize: '0.74rem', fontWeight: 700, color: info?.gpu_available ? 'var(--cyan-light)' : 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {info?.gpu_available ? 'NVIDIA CUDA Acceleration Online' : 'CPU Execution Engine Active'}
              </span>
            </div>
            <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
              Hardware & Execution Telemetry
            </h2>
          </div>

          <button onClick={loadInfo} className="neuro-btn neuro-btn-sm">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
              <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
              <path d="M8 16H3v5" />
            </svg>
            <span>Refresh Telemetry</span>
          </button>
        </div>
      </div>

      {/* GPU & Hardware Specs Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
        <div className="neuro-card" style={{ padding: '1.4rem' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Primary Device
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', marginTop: '0.4rem', fontFamily: 'var(--font-brand)' }}>
            {info?.gpu_name || 'CPU (Fallback Engine)'}
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
            Execution Device: <strong style={{ color: 'var(--cyan-light)', fontFamily: 'var(--font-mono)' }}>{info?.execution_device.toUpperCase()}</strong>
          </div>
        </div>

        <div className="neuro-card" style={{ padding: '1.4rem' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            VRAM Capacity
          </div>
          <div className="metric-value" style={{ fontSize: '1.5rem', color: info?.gpu_available ? 'var(--cyan-light)' : 'var(--text-muted)', marginTop: '0.4rem' }}>
            {info?.vram_total_gb ? `${info.vram_total_gb} GB` : 'N/A (Host RAM)'}
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
            Allocated: {info?.vram_allocated_gb ? `${info.vram_allocated_gb} GB` : '0 GB'}
          </div>
        </div>

        <div className="neuro-card" style={{ padding: '1.4rem' }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            PyTorch CUDA Runtime
          </div>
          <div className="metric-value" style={{ fontSize: '1.5rem', color: 'var(--text-primary)', marginTop: '0.4rem' }}>
            {info?.cuda_version ? `CUDA ${info.cuda_version}` : 'CPU Native'}
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
            PyTorch v{info?.pytorch_version}
          </div>
        </div>
      </div>

      {/* Host System Environment Table */}
      <div className="neuro-card" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '1rem' }}>
          Compute Environment Specifications
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Operating System Platform:</span>
            <strong style={{ color: 'var(--text-primary)' }}>{info?.os_name}</strong>
          </div>

          <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Python Version:</span>
            <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{info?.python_version}</strong>
          </div>

          <div className="neuro-sunken" style={{ padding: '0.85rem 1.1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>CPU Fallback Status:</span>
            <strong style={{ color: info?.cpu_fallback_active ? 'var(--amber-light)' : 'var(--emerald-light)' }}>
              {info?.cpu_fallback_active ? 'Active (CPU Safe Fallback)' : 'Inactive (GPU Acceleration Direct)'}
            </strong>
          </div>
        </div>

        <div className="neuro-sunken" style={{ padding: '1rem', marginTop: '1.25rem', fontSize: '0.78rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Scientific Note:</strong> GPU acceleration speeds up DistilBERT batch representations and neural classifier retraining while strictly preserving mathematical reproducibility and seed determinism.
        </div>
      </div>
    </div>
  );
};
