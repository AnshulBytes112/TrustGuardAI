import React from 'react';

interface EmptyStateProps {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  actionText,
  onAction,
  icon,
}) => {
  return (
    <div
      className="neuro-card"
      style={{
        padding: '3.5rem 2rem',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '1.5rem 0',
      }}
    >
      <div
        style={{
          width: '56px',
          height: '56px',
          borderRadius: '16px',
          backgroundColor: 'var(--bg-sunken)',
          border: '1px solid var(--border-default)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          marginBottom: '1.25rem',
        }}
      >
        {icon || (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        )}
      </div>

      <h3
        style={{
          fontFamily: 'var(--font-brand)',
          fontSize: '1.15rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '0.4rem',
        }}
      >
        {title}
      </h3>

      <p
        style={{
          fontSize: '0.86rem',
          color: 'var(--text-dim)',
          maxWidth: '460px',
          lineHeight: 1.5,
          marginBottom: actionText ? '1.5rem' : '0',
        }}
      >
        {description}
      </p>

      {actionText && onAction && (
        <button onClick={onAction} className="neuro-btn neuro-btn-primary">
          {actionText}
        </button>
      )}
    </div>
  );
};
