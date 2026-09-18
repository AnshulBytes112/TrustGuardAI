import React from 'react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  details?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'System Operation Failed',
  message,
  onRetry,
  details,
}) => {
  return (
    <div
      className="neuro-card"
      style={{
        padding: '2.5rem 2rem',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid var(--rose-border)',
        backgroundColor: 'rgba(244, 63, 94, 0.04)',
        margin: '1.5rem 0',
      }}
    >
      <div
        style={{
          width: '50px',
          height: '50px',
          borderRadius: '14px',
          backgroundColor: 'var(--rose-dim)',
          border: '1px solid var(--rose-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--rose-light)',
          marginBottom: '1rem',
        }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>

      <h3
        style={{
          fontFamily: 'var(--font-brand)',
          fontSize: '1.1rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '0.35rem',
        }}
      >
        {title}
      </h3>

      <p
        style={{
          fontSize: '0.86rem',
          color: 'var(--text-secondary)',
          maxWidth: '520px',
          lineHeight: 1.5,
          marginBottom: details || onRetry ? '1.25rem' : '0',
        }}
      >
        {message}
      </p>

      {details && (
        <pre
          className="neuro-sunken"
          style={{
            padding: '0.75rem 1rem',
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono)',
            color: 'var(--rose-light)',
            maxWidth: '650px',
            width: '100%',
            overflowX: 'auto',
            textAlign: 'left',
            marginBottom: onRetry ? '1.25rem' : '0',
          }}
        >
          {details}
        </pre>
      )}

      {onRetry && (
        <button onClick={onRetry} className="neuro-btn neuro-btn-danger neuro-btn-sm">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M8 16H3v5" />
          </svg>
          <span>Retry Operation</span>
        </button>
      )}
    </div>
  );
};
