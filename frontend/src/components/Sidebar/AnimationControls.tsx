import React, { useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';

export const AnimationControls: React.FC = () => {
  const {
    animationState,
    setAnimationState,
    currentStep,
    totalSteps,
    setCurrentStep,
    speed,
    setSpeed,
    trajectories,
  } = useAppContext();

  const hasTrajectories = Object.keys(trajectories).length > 0;

  const handlePlayPause = useCallback(() => {
    if (animationState === 'playing') {
      setAnimationState('paused');
    } else if (animationState === 'paused') {
      setAnimationState('playing');
    } else {
      // stopped - start playing
      setAnimationState('playing');
    }
  }, [animationState, setAnimationState]);

  const handleStop = useCallback(() => {
    setAnimationState('stopped');
    setCurrentStep(0);
  }, [setAnimationState, setCurrentStep]);

  const handleTimelineChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setCurrentStep(parseInt(e.target.value, 10));
    },
    [setCurrentStep]
  );

  const handleSpeedChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSpeed(parseFloat(e.target.value));
    },
    [setSpeed]
  );

  return (
    <>
      {/* Media Controls */}
      <div className="media-controls">
        <button
          id="play-pause-btn"
          className={`media-btn ${animationState === 'stopped' ? 'primary' : ''}`}
          title={animationState === 'playing' ? 'Pause' : 'Play'}
          onClick={handlePlayPause}
          disabled={!hasTrajectories}
        >
          <svg
            className={`icon play-icon ${animationState === 'playing' ? 'hidden' : ''}`}
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          <svg
            className={`icon pause-icon ${animationState === 'playing' ? '' : 'hidden'}`}
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
          </svg>
        </button>
        <button
          id="stop-btn"
          className="media-btn secondary"
          title="Stop"
          onClick={handleStop}
          disabled={animationState === 'stopped'}
        >
          <svg className="icon stop-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <rect x="6" y="6" width="12" height="12" />
          </svg>
        </button>
      </div>

      {/* Timeline Controls Container */}
      <div className="timeline-container">
        {/* Timeline Scrubber */}
        <div className="slider-group">
          <div className="slider-label">
            <span>Iteration</span>
            <span className="slider-value">
              <span id="current-step">{currentStep}</span>
              <span className="separator">/</span>
              <span id="total-steps">{totalSteps}</span>
            </span>
          </div>
          <input
            type="range"
            id="timeline-scrubber"
            className="timeline-scrubber"
            min="0"
            max={totalSteps}
            value={currentStep}
            onChange={handleTimelineChange}
            disabled={!hasTrajectories}
          />
        </div>

        {/* Speed Slider */}
        <div className="slider-group">
          <div className="slider-label">
            <span>Speed</span>
            <span id="speed-value" className="slider-value">
              {speed.toFixed(1)}
            </span>
          </div>
          <input
            type="range"
            id="speed"
            className="timeline-scrubber"
            min="0.1"
            max="5"
            step="0.1"
            value={speed}
            onChange={handleSpeedChange}
          />
        </div>
      </div>
    </>
  );
};
