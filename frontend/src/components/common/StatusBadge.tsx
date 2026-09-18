import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'sm' }) => {
  const norm = status.toUpperCase();

  let tone: 'emerald' | 'cyan' | 'amber' | 'rose' | 'violet' = 'cyan';
  let pipTone = 'cyan';
  let label = status;

  if (norm === 'COMPLETED' || norm === 'SUCCESS' || norm === 'ACTIVE' || norm === 'RETAIN' || norm === 'CLEAN') {
    tone = 'emerald';
    pipTone = 'emerald';
    label = norm === 'RETAIN' ? 'Clean Retained' : status;
  } else if (norm === 'RUNNING' || norm === 'PROCESSING' || norm === 'CALIBRATING') {
    tone = 'cyan';
    pipTone = 'cyan';
  } else if (norm === 'PENDING' || norm === 'CREATED' || norm === 'WAITING') {
    tone = 'amber';
    pipTone = 'amber';
  } else if (norm === 'FAILED' || norm === 'ERROR' || norm === 'ISOLATE' || norm === 'QUARANTINED' || norm === 'POISONED') {
    tone = 'rose';
    pipTone = 'rose';
    label = norm === 'ISOLATE' ? 'Poison Quarantined' : status;
  } else if (norm === 'RESTORED') {
    tone = 'violet';
    pipTone = 'violet';
  }

  const isAnimated = norm === 'RUNNING' || norm === 'PROCESSING';

  return (
    <span
      className={`neuro-badge ${tone}`}
      style={{
        fontSize: size === 'sm' ? '0.7rem' : '0.78rem',
        padding: size === 'sm' ? '0.2rem 0.55rem' : '0.35rem 0.75rem',
      }}
    >
      <span className={`status-pip ${pipTone} ${isAnimated ? pipTone : ''}`} />
      <span>{label}</span>
    </span>
  );
};
