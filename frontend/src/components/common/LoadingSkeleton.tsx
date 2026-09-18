import React from 'react';

interface LoadingSkeletonProps {
  lines?: number;
  height?: string;
  type?: 'card' | 'table' | 'text';
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  lines = 4,
  height = '180px',
  type = 'card',
}) => {
  if (type === 'card') {
    return (
      <div
        className="neuro-card"
        style={{
          padding: '1.5rem',
          minHeight: height,
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          animation: 'pulse-opacity 1.5s infinite ease-in-out',
        }}
      >
        <div style={{ height: '18px', width: '40%', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-xs)' }} />
        <div style={{ height: '36px', width: '70%', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-xs)' }} />
        <div style={{ height: '14px', width: '55%', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-xs)' }} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1rem 0' }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          style={{
            height: '24px',
            width: `${100 - (i % 3) * 15}%`,
            backgroundColor: 'var(--bg-surface-elevated)',
            borderRadius: 'var(--radius-xs)',
            animation: 'pulse-opacity 1.5s infinite ease-in-out',
          }}
        />
      ))}
    </div>
  );
};
