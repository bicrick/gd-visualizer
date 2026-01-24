import React, { useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';

export const DisplayOptions: React.FC = () => {
  const { showTrails, setShowTrails } = useAppContext();

  const handleShowTrailsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setShowTrails(e.target.checked);
    },
    [setShowTrails]
  );

  return (
    <div className="control-item">
      <label>
        <input
          type="checkbox"
          id="show-trails"
          checked={showTrails}
          onChange={handleShowTrailsChange}
        />
        Show Trajectory Trails
      </label>
    </div>
  );
};
