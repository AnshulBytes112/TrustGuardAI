import React, { useEffect, useState } from 'react';
import { fetchDatasets, fetchDatasetDetail, uploadDatasetFile } from '../../api';
import type { DatasetItem, DatasetDetail, SampleItem } from '../../api';
import { StatusBadge } from '../common/StatusBadge';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';
import { EmptyState } from '../common/EmptyState';

interface DatasetsViewProps {
  onLaunchInvestigation: (datasetId: string) => void;
  searchQuery?: string;
}

export const DatasetsView: React.FC<DatasetsViewProps> = ({
  onLaunchInvestigation,
  searchQuery = '',
}) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedDatasetDetail, setSelectedDatasetDetail] = useState<DatasetDetail | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  
  // Upload State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [customName, setCustomName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  // Filter & Pagination State
  const [splitFilter, setSplitFilter] = useState<'ALL' | 'TRAIN' | 'VALIDATION' | 'TEST'>('ALL');
  const [stateFilter] = useState<'ALL' | 'ACTIVE' | 'QUARANTINED' | 'RESTORED'>('ALL');
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDatasets = React.useCallback(async (selectFirst = true) => {
    try {
      setLoading(true);
      setError(null);
      const list = await fetchDatasets();
      setDatasets(list);
      if (list.length > 0 && selectFirst && !selectedDatasetId) {
        setSelectedDatasetId(list[0].id);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  }, [selectedDatasetId]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    const loadDetail = async () => {
      try {
        setDetailLoading(true);
        const detail = await fetchDatasetDetail(selectedDatasetId);
        setSelectedDatasetDetail(detail);
        setSamples(detail.samples || []);
      } catch (err: any) {
        console.error(err);
      } finally {
        setDetailLoading(false);
      }
    };
    loadDetail();
  }, [selectedDatasetId]);

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    try {
      setUploading(true);
      setUploadError(null);
      setUploadSuccess(false);

      const created = await uploadDatasetFile(uploadFile, customName.trim() || undefined);
      setUploadSuccess(true);
      setUploadFile(null);
      setCustomName('');
      await loadDatasets(false);
      setSelectedDatasetId(created.id);
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const filteredDatasets = datasets.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredSamples = samples.filter((s) => {
    if (splitFilter !== 'ALL' && s.split !== splitFilter) return false;
    if (stateFilter !== 'ALL' && s.state !== stateFilter) return false;
    return true;
  });

  const totalPages = Math.ceil(filteredSamples.length / pageSize) || 1;
  const paginatedSamples = filteredSamples.slice((page - 1) * pageSize, page * pageSize);

  if (loading && datasets.length === 0) {
    return <LoadingSkeleton type="table" lines={8} />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => loadDatasets(true)} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Upload Ingestion Drawer / Card */}
      <div className="neuro-card" style={{ padding: '1.5rem 1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>
              Dataset Ingestion Engine
            </h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
              Supports .jsonl, .csv, .txt, .json with deterministic schema adaptation and train/val/test splitting
            </p>
          </div>
        </div>

        <form onSubmit={handleFileUpload} style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 240px' }}>
            <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              DATASET FILE (.CSV, .JSONL, .TXT)
            </label>
            <input
              type="file"
              accept=".csv,.jsonl,.txt,.json"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setUploadFile(e.target.files[0]);
                  if (!customName) {
                    setCustomName(e.target.files[0].name.replace(/\.[^/.]+$/, ''));
                  }
                }
              }}
              className="neuro-input"
              style={{ padding: '0.45rem 0.75rem', fontSize: '0.8rem' }}
            />
          </div>

          <div style={{ flex: '1 1 200px' }}>
            <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              DATASET ALIAS / NAME
            </label>
            <input
              type="text"
              placeholder="e.g. sst2_benchmark_v1"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="neuro-input"
              style={{ fontSize: '0.8rem' }}
            />
          </div>

          <button
            type="submit"
            disabled={!uploadFile || uploading}
            className="neuro-btn neuro-btn-primary"
            style={{ padding: '0.62rem 1.4rem' }}
          >
            {uploading ? (
              <>
                <span className="status-pip cyan" />
                <span>Ingesting Corpus...</span>
              </>
            ) : (
              <>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span>Upload & Ingest</span>
              </>
            )}
          </button>
        </form>

        {uploadError && (
          <div style={{ marginTop: '0.9rem', fontSize: '0.78rem', color: 'var(--rose-light)' }}>
            Error: {uploadError}
          </div>
        )}
        {uploadSuccess && (
          <div style={{ marginTop: '0.9rem', fontSize: '0.78rem', color: 'var(--emerald-light)' }}>
            Dataset successfully ingested and indexed into TrustGuardAI storage.
          </div>
        )}
      </div>

      {/* Main Two-Pane Explorer */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '1.5rem' }}>
        {/* Left: Datasets List */}
        <div className="neuro-card" style={{ padding: '1.25rem', height: 'fit-content' }}>
          <div style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-dim)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
            Ingested Corpora ({filteredDatasets.length})
          </div>

          {filteredDatasets.length === 0 ? (
            <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.82rem' }}>
              No datasets matching query.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '560px', overflowY: 'auto' }}>
              {filteredDatasets.map((d) => {
                const isSelected = selectedDatasetId === d.id;
                return (
                  <div
                    key={d.id}
                    onClick={() => setSelectedDatasetId(d.id)}
                    className={isSelected ? 'neuro-card' : 'neuro-sunken'}
                    style={{
                      padding: '0.85rem 1rem',
                      cursor: 'pointer',
                      border: isSelected ? '1px solid var(--cyan-border)' : '1px solid var(--border-subtle)',
                      backgroundColor: isSelected ? 'var(--bg-surface-elevated)' : 'var(--bg-sunken)',
                      transition: 'all var(--transition-fast)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.88rem', color: isSelected ? '#ffffff' : 'var(--text-primary)' }}>
                        {d.name}
                      </span>
                      <span className="neuro-badge cyan" style={{ fontSize: '0.65rem' }}>
                        {d.modality}
                      </span>
                    </div>

                    <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                      {d.total_samples} samples &bull; {d.label_mode}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Selected Dataset Detail & Paginated Samples */}
        <div className="neuro-card" style={{ padding: '1.5rem' }}>
          {selectedDatasetDetail ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Header Info */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
                      {selectedDatasetDetail.name}
                    </h2>
                    <span className="code-badge">{selectedDatasetDetail.id.substring(0, 8)}</span>
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                    Created: {new Date(selectedDatasetDetail.created_at).toLocaleString()} &bull; Source: {selectedDatasetDetail.source}
                  </div>
                </div>

                <button
                  onClick={() => onLaunchInvestigation(selectedDatasetDetail.id)}
                  className="neuro-btn neuro-btn-primary neuro-btn-sm"
                  style={{ fontWeight: 700 }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Investigate This Dataset</span>
                </button>
              </div>

              {/* Split Metrics Tiles */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.9rem' }}>
                <div className="neuro-sunken" style={{ padding: '0.85rem 1rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>TOTAL SAMPLES</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>
                    {selectedDatasetDetail.total_samples}
                  </div>
                </div>

                <div className="neuro-sunken" style={{ padding: '0.85rem 1rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--emerald-light)', fontWeight: 600 }}>TRAIN SPLIT</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--emerald-light)', marginTop: '0.2rem' }}>
                    {selectedDatasetDetail.train_count}
                  </div>
                </div>

                <div className="neuro-sunken" style={{ padding: '0.85rem 1rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--cyan-light)', fontWeight: 600 }}>VALIDATION SPLIT</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--cyan-light)', marginTop: '0.2rem' }}>
                    {selectedDatasetDetail.val_count}
                  </div>
                </div>

                <div className="neuro-sunken" style={{ padding: '0.85rem 1rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--violet-light)', fontWeight: 600 }}>TEST SPLIT</div>
                  <div className="metric-value" style={{ fontSize: '1.35rem', color: 'var(--violet-light)', marginTop: '0.2rem' }}>
                    {selectedDatasetDetail.test_count}
                  </div>
                </div>
              </div>

              {/* Filter Controls */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.5rem' }}>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {(['ALL', 'TRAIN', 'VALIDATION', 'TEST'] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setSplitFilter(s);
                        setPage(1);
                      }}
                      className="neuro-btn neuro-btn-sm"
                      style={{
                        backgroundColor: splitFilter === s ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
                        borderColor: splitFilter === s ? 'var(--cyan-border)' : 'var(--border-subtle)',
                        color: splitFilter === s ? 'var(--cyan-light)' : 'var(--text-muted)',
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>

                <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                  Showing {paginatedSamples.length} of {filteredSamples.length} samples
                </div>
              </div>

              {/* Sample Table */}
              {detailLoading ? (
                <LoadingSkeleton type="table" lines={6} />
              ) : paginatedSamples.length === 0 ? (
                <EmptyState title="No Samples Matching Filter" description="Try selecting a different split or upload more dataset records." />
              ) : (
                <div className="neuro-table-wrapper">
                  <table className="neuro-table">
                    <thead>
                      <tr>
                        <th style={{ width: '80px' }}>Index</th>
                        <th>Text Content</th>
                        <th style={{ width: '100px' }}>Label</th>
                        <th style={{ width: '90px' }}>Split</th>
                        <th style={{ width: '110px' }}>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedSamples.map((s, idx) => (
                        <tr key={s.id || idx}>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                            {(page - 1) * pageSize + idx + 1}
                          </td>
                          <td style={{ maxWidth: '420px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={s.text}>
                            {s.text}
                          </td>
                          <td>
                            <span className="code-badge">{s.label || 'None'}</span>
                          </td>
                          <td>
                            <span className="neuro-badge" style={{ fontSize: '0.65rem' }}>
                              {s.split}
                            </span>
                          </td>
                          <td>
                            <StatusBadge status={s.state} size="sm" />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Pagination */}
              {totalPages > 1 && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <button
                    disabled={page === 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    className="neuro-btn neuro-btn-sm"
                  >
                    Previous
                  </button>
                  <span style={{ display: 'flex', alignItems: 'center', fontSize: '0.76rem', color: 'var(--text-dim)', padding: '0 0.5rem' }}>
                    Page {page} of {totalPages}
                  </span>
                  <button
                    disabled={page === totalPages}
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    className="neuro-btn neuro-btn-sm"
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          ) : (
            <EmptyState title="No Dataset Selected" description="Select a dataset from the list on the left to inspect its samples." />
          )}
        </div>
      </div>
    </div>
  );
};
