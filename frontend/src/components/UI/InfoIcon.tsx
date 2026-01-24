import React from 'react';

interface InfoIconProps {
  tooltip: string;
}

export const InfoIcon: React.FC<InfoIconProps> = ({ tooltip }) => {
  return (
    <i className="info-icon" data-tooltip={tooltip}>
      i
    </i>
  );
};
