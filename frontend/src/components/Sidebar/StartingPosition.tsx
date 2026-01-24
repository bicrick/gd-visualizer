import React, { useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';

export const StartingPosition: React.FC = () => {
  const { startPosition, setStartPosition, randomizeStartPosition, setPickingMode, pickingMode } =
    useAppContext();

  const handleXChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setStartPosition(parseFloat(e.target.value), startPosition.y);
    },
    [setStartPosition, startPosition.y]
  );

  const handleYChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setStartPosition(startPosition.x, parseFloat(e.target.value));
    },
    [setStartPosition, startPosition.x]
  );

  const handleRandomStart = useCallback(() => {
    randomizeStartPosition();
  }, [randomizeStartPosition]);

  const handlePickPoint = useCallback(() => {
    setPickingMode(!pickingMode);
  }, [setPickingMode, pickingMode]);

  return (
    <>
      <div className="control-item">
        <label htmlFor="start-x">X:</label>
        <input
          type="number"
          id="start-x"
          min="-5"
          max="5"
          step="0.1"
          value={startPosition.x}
          onChange={handleXChange}
        />
      </div>
      <div className="control-item">
        <label htmlFor="start-y">Y:</label>
        <input
          type="number"
          id="start-y"
          min="-5"
          max="5"
          step="0.1"
          value={startPosition.y}
          onChange={handleYChange}
        />
      </div>
      <div className="button-group">
        <button
          id="random-start"
          onClick={handleRandomStart}
          data-tooltip="Randomly selects a starting position on the loss landscape for optimization."
        >
          Random Start
        </button>
        <button
          id="pick-point-btn"
          onClick={handlePickPoint}
          data-tooltip="Click on the loss landscape to select your starting initial conditions."
          className={pickingMode ? 'active' : ''}
        >
          Pick Point
        </button>
      </div>
    </>
  );
};
