import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badgeText?: string;
  badgeTone?: 'emerald' | 'cyan' | 'amber' | 'rose' | 'violet';
  icon?: React.ReactNode;
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'neutral';
    isPositiveGood?: boolean;
  };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badgeText,
  badgeTone = 'cyan',
  icon,
  trend,
}) => {
  return (
    <div
      className="neuro-card"
      style={{
        padding: '1.2rem 1.4rem',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.01em' }}>
          {title}
        </span>
        {icon && (
          <div
            style={{
              color: 'var(--text-dim)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.35rem' }}>
        <div
          className="metric-value"
          style={{
            fontSize: '1.65rem',
            fontWeight: 800,
            color: 'var(--text-primary)',
            lineHeight: 1.1,
          }}
        >
          {value}
        </div>
        {badgeText && (
          <span className={`neuro-badge ${badgeTone}`} style={{ fontSize: '0.68rem', padding: '0.15rem 0.5rem' }}>
            {badgeText}
          </span>
        )}
      </div>

      {(subtitle || trend) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          {trend && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.2rem',
                fontWeight: 600,
                color:
                  trend.direction === 'neutral'
                    ? 'var(--text-muted)'
                    : (trend.direction === 'up' && trend.isPositiveGood !== false) || (trend.direction === 'down' && trend.isPositiveGood === false)
                    ? 'var(--emerald)'
                    : 'var(--rose)',
              }}
            >
              {trend.direction === 'up' && '▲'}
              {trend.direction === 'down' && '▼'}
              {trend.value}
            </span>
          )}
          {subtitle && <span>{subtitle}</span>}
        </div>
      )}
    </div>
  );
};
