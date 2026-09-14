import React from 'react';

interface BadgeProps {
  type: 'risk' | 'status' | 'label' | 'custom';
  value: string;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ type, value, className = '' }) => {
  let badgeClass = 'badge-queued';

  const valUpper = value.toUpperCase();

  if (valUpper === 'CRITICAL' || valUpper === 'ILLICIT') {
    badgeClass = 'badge-critical';
  } else if (valUpper === 'HIGH' || valUpper === 'ESCALATED') {
    badgeClass = 'badge-high';
  } else if (valUpper === 'MEDIUM' || valUpper === 'INCONCLUSIVE') {
    badgeClass = 'badge-medium';
  } else if (valUpper === 'LOW' || valUpper === 'LICIT') {
    badgeClass = 'badge-low';
  } else if (valUpper === 'REVIEWED') {
    badgeClass = 'badge-reviewed';
  } else if (valUpper === 'QUEUED' || valUpper === 'PENDING') {
    badgeClass = 'badge-queued';
  }

  return (
    <span className={`badge ${badgeClass} ${className}`}>
      {value}
    </span>
  );
};
