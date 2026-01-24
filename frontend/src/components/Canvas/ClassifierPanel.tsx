import React, { useEffect, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { api } from '../../services/api';
import { ClassifierDataPoint } from '../../context/types';

export const ClassifierPanel: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const datasetRef = useRef<ClassifierDataPoint[]>([]);
  const { currentManifoldId, trajectories, currentStep, enabledOptimizers } = useAppContext();

  const isClassifierManifold = currentManifoldId === 'neural_net_classifier';

  // Load classifier dataset
  useEffect(() => {
    if (!isClassifierManifold) return;

    const loadDataset = async () => {
      try {
        const data = await api.getClassifierDataset();
        datasetRef.current = data.points;
        renderClassifierViz();
      } catch (error) {
        console.error('Error loading classifier dataset:', error);
      }
    };

    loadDataset();
  }, [isClassifierManifold]);

  // Render visualization
  const renderClassifierViz = (optimizerPositions?: Record<string, { x: number; y: number }>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw decision boundary (if optimizer positions provided)
    if (optimizerPositions) {
      Object.entries(optimizerPositions).forEach(([name, pos]) => {
        if (!enabledOptimizers[name as keyof typeof enabledOptimizers]) return;

        // Draw decision boundary circle
        ctx.beginPath();
        ctx.arc(
          dataToCanvas(pos.x, -2, 2, width),
          dataToCanvas(pos.y, -2, 2, height),
          width / 8,
          0,
          2 * Math.PI
        );
        ctx.strokeStyle = getOptimizerColor(name);
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    // Draw dataset points
    datasetRef.current.forEach((point) => {
      ctx.beginPath();
      ctx.arc(
        dataToCanvas(point.x, -2, 2, width),
        dataToCanvas(point.y, -2, 2, height),
        3,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = point.label === 0 ? '#4444ff' : '#ff4444';
      ctx.fill();
    });
  };

  // Update visualization when trajectories or step changes
  useEffect(() => {
    if (!isClassifierManifold || Object.keys(trajectories).length === 0) return;

    const positions: Record<string, { x: number; y: number }> = {};
    Object.entries(trajectories).forEach(([name, trajectory]) => {
      if (trajectory && currentStep < trajectory.length) {
        const point = trajectory[currentStep];
        positions[name] = { x: point.x, y: point.y };
      }
    });

    renderClassifierViz(positions);
  }, [isClassifierManifold, trajectories, currentStep, enabledOptimizers]);

  const dataToCanvas = (value: number, min: number, max: number, canvasSize: number) => {
    return ((value - min) / (max - min)) * canvasSize;
  };

  const getOptimizerColor = (name: string): string => {
    const colors: Record<string, string> = {
      sgd: '#ff4444',
      batch: '#4444ff',
      momentum: '#44ff44',
      adam: '#ff8800',
      ballistic: '#ff00ff',
      ballistic_adam: '#00ffff',
    };
    return colors[name] || '#ffffff';
  };

  if (!isClassifierManifold) return null;

  return (
    <div id="classifier-panel">
      <div className="classifier-header">
        <h3>Two-Circle Clustering</h3>
        <p>Two clusters - which one will it find?</p>
      </div>
      <canvas ref={canvasRef} id="classifier-canvas" width={300} height={300}></canvas>
    </div>
  );
};
