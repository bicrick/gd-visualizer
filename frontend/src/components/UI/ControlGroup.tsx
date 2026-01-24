import React, { ReactNode } from 'react';

interface ControlGroupProps {
  title: string;
  children: ReactNode;
}

export const ControlGroup: React.FC<ControlGroupProps> = ({ title, children }) => {
  return (
    <div className="control-group">
      <h2>{title}</h2>
      {children}
    </div>
  );
};
