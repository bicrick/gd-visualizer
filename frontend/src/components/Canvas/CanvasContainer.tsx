import React, { useRef, useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import { useThreeScene } from '../../hooks/useThreeScene';
import { useOptimizers } from '../../hooks/useOptimizers';
import { useAnimation } from '../../hooks/useAnimation';
import { ManifoldSelector } from './ManifoldSelector';
import { ManifoldParamsPanel } from './ManifoldParamsPanel';
import { ClassifierPanel } from './ClassifierPanel';
import { ThemeToggle } from './ThemeToggle';

export const CanvasContainer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { startPosition } = useAppContext();

  const { scene, isLoading, landscapeZRange, updateStartPointMarker } = useThreeScene(canvasRef);
  const { updateBallPositions } = useOptimizers(scene, landscapeZRange);
  useAnimation(updateBallPositions);

  // Update start point marker when position changes
  useEffect(() => {
    if (updateStartPointMarker) {
      updateStartPointMarker(startPosition.x, startPosition.y);
    }
  }, [startPosition, updateStartPointMarker]);

  return (
    <div id="canvas-container">
      <canvas ref={canvasRef} id="canvas"></canvas>
      <div id="loading" className={isLoading ? '' : 'hidden'}>
        Loading landscape...
      </div>
      <ManifoldSelector />
      <ManifoldParamsPanel />
      <ClassifierPanel />
      <ThemeToggle />
    </div>
  );
};
