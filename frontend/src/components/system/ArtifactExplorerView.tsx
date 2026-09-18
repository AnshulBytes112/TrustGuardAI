import React, { useEffect, useState } from 'react';
import { fetchResearchArtifacts, fetchResearchArtifactContent } from '../../api';
import type { ResearchArtifactItem } from '../../api';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ErrorState } from '../common/ErrorState';
import { EmptyState } from '../common/EmptyState';

export const ArtifactExplorerView: React.FC = () => {
  const [artifacts, setArtifacts] = useState<ResearchArtifactItem[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState<ResearchArtifactItem | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('ALL');

  const loadArtifacts = async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await fetchResearchArtifacts();
      setArtifacts(list);
      if (list.length > 0) {
        setSelectedArtifact(list[0]);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load research artifacts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadArtifacts();
  }, []);

  useEffect(() => {
    if (!selectedArtifact) {
      setPreviewContent(null);
      return;
    }
    const loadPreview = async () => {
      try {
        setPreviewLoading(true);
        const text = await fetchResearchArtifactContent(selectedArtifact.relative_path);
        setPreviewContent(text);
      } catch (err: any) {
        setPreviewContent(`Failed to preview artifact: ${err.message}`);
      } finally {
        setPreviewLoading(false);
      }
    };
    loadPreview();
  }, [selectedArtifact]);

  if (loading && artifacts.length === 0) {
    return <LoadingSkeleton type="table" lines={8} />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={loadArtifacts} />;
  }

  const categories = ['ALL', ...Array.from(new Set(artifacts.map((a) => a.category)))];
  const filtered = artifacts.filter((a) => categoryFilter === 'ALL' || a.category === categoryFilter);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header Info */}
      <div className="neuro-card" style={{ padding: '1.5rem 1.75rem' }}>
        <h2 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.35rem', fontWeight: 800, color: '#ffffff' }}>
          Research Artifacts & Empirical Data Browser
        </h2>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', maxWidth: '820px', marginTop: '0.35rem' }}>
          Persistent empirical evaluation datasets, sweep telemetry, ablation results, and serialized model metrics.
        </p>
      </div>

      {/* Category Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto' }}>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setCategoryFilter(cat)}
            className="neuro-btn neuro-btn-sm"
            style={{
              backgroundColor: categoryFilter === cat ? 'var(--cyan-dim)' : 'var(--bg-sunken)',
              borderColor: categoryFilter === cat ? 'var(--cyan-border)' : 'var(--border-subtle)',
              color: categoryFilter === cat ? 'var(--cyan-light)' : 'var(--text-muted)',
              textTransform: 'capitalize',
            }}
          >
            {cat.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      {/* Two Pane Browser */}
      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '1.5rem' }}>
        {/* Left: Artifact List */}
        <div className="neuro-card" style={{ padding: '1.25rem', height: 'fit-content' }}>
          <div style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-dim)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
            Files ({filtered.length})
          </div>

          {filtered.length === 0 ? (
            <EmptyState title="No Artifacts" description="No research artifacts found for this category." />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '580px', overflowY: 'auto' }}>
              {filtered.map((art) => {
                const isSelected = selectedArtifact?.relative_path === art.relative_path;
                return (
                  <div
                    key={art.relative_path}
                    onClick={() => setSelectedArtifact(art)}
                    className={isSelected ? 'neuro-card' : 'neuro-sunken'}
                    style={{
                      padding: '0.85rem 1rem',
                      cursor: 'pointer',
                      border: isSelected ? '1px solid var(--cyan-border)' : '1px solid var(--border-subtle)',
                      backgroundColor: isSelected ? 'var(--bg-surface-elevated)' : 'var(--bg-sunken)',
                      transition: 'all var(--transition-fast)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.86rem', color: isSelected ? '#ffffff' : 'var(--text-primary)' }}>
                        {art.name}
                      </span>
                      <span className="code-badge" style={{ fontSize: '0.65rem' }}>
                        {art.file_type}
                      </span>
                    </div>

                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                      {(art.size_bytes / 1024).toFixed(1)} KB &bull; {art.category}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Content Preview */}
        <div className="neuro-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', minHeight: '480px' }}>
          {selectedArtifact ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.75rem' }}>
                <div>
                  <h3 style={{ fontFamily: 'var(--font-brand)', fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>
                    {selectedArtifact.name}
                  </h3>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    artifacts/research/{selectedArtifact.relative_path}
                  </span>
                </div>

                <span className="neuro-badge cyan">
                  {(selectedArtifact.size_bytes / 1024).toFixed(1)} KB
                </span>
              </div>

              {previewLoading ? (
                <LoadingSkeleton type="text" lines={10} />
              ) : (
                <pre
                  className="neuro-sunken"
                  style={{
                    flex: 1,
                    padding: '1rem 1.25rem',
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-secondary)',
                    overflow: 'auto',
                    maxHeight: '520px',
                    lineHeight: 1.5,
                  }}
                >
                  {previewContent}
                </pre>
              )}
            </>
          ) : (
            <EmptyState title="No Artifact Selected" description="Choose an artifact from the list to preview its data." />
          )}
        </div>
      </div>
    </div>
  );
};
