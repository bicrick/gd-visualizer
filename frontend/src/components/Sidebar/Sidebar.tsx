import React from 'react';
import { ControlGroup } from '../UI/ControlGroup';
import { StartingPosition } from './StartingPosition';
import { AnimationControls } from './AnimationControls';
import { DisplayOptions } from './DisplayOptions';

interface SidebarProps {
  children?: React.ReactNode;
}

export const Sidebar: React.FC<SidebarProps> = ({ children }) => {
  return (
    <div id="sidebar">
      <div className="sidebar-header">
        <h1>Gradient Descent Visualizer</h1>
      </div>

      <div className="sidebar-content">
        <ControlGroup title="Starting Position">
          <StartingPosition />
        </ControlGroup>

        <ControlGroup title="Animation">
          <AnimationControls />
        </ControlGroup>

        <ControlGroup title="Display Options">
          <DisplayOptions />
        </ControlGroup>

        {children}
      </div>
    </div>
  );
};
