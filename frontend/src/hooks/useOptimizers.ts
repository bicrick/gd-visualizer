import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useAppContext } from '../context/AppContext';
import { OptimizerName } from '../context/types';

const OPTIMIZER_COLORS: Record<OptimizerName, number> = {
  sgd: 0xff4444,
  batch: 0x4444ff,
  momentum: 0x44ff44,
  adam: 0xff8800,
  ballistic: 0xff00ff,
  ballistic_adam: 0x00ffff,
};

interface OptimizerRefs {
  balls: Record<OptimizerName, THREE.Mesh>;
  lines: Record<OptimizerName, THREE.Line>;
}

export const useOptimizers = (
  scene: THREE.Scene | null,
  landscapeZRange: { zMin: number; zMax: number; scale: number }
) => {
  const refsRef = useRef<OptimizerRefs | null>(null);
  const { trajectories, enabledOptimizers, showTrails } = useAppContext();

  // Initialize optimizer balls and lines
  useEffect(() => {
    if (!scene) return;

    const balls: Record<string, THREE.Mesh> = {};
    const lines: Record<string, THREE.Line> = {};

    Object.entries(OPTIMIZER_COLORS).forEach(([name, color]) => {
      // Create ball
      const ballGeometry = new THREE.SphereGeometry(0.15, 16, 16);
      const ballMaterial = new THREE.MeshPhongMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.3,
      });
      const ball = new THREE.Mesh(ballGeometry, ballMaterial);
      ball.visible = false;
      scene.add(ball);
      balls[name] = ball;

      // Create line
      const lineMaterial = new THREE.LineBasicMaterial({
        color,
        linewidth: 2,
        opacity: 0.6,
        transparent: true,
      });
      const lineGeometry = new THREE.BufferGeometry();
      const line = new THREE.Line(lineGeometry, lineMaterial);
      line.visible = false;
      scene.add(line);
      lines[name] = line;
    });

    refsRef.current = { balls: balls as any, lines: lines as any };

    return () => {
      // Cleanup
      Object.values(balls).forEach((ball) => {
        scene.remove(ball);
        ball.geometry.dispose();
        if (Array.isArray(ball.material)) {
          ball.material.forEach((m) => m.dispose());
        } else {
          ball.material.dispose();
        }
      });
      Object.values(lines).forEach((line) => {
        scene.remove(line);
        line.geometry.dispose();
        if (Array.isArray(line.material)) {
          line.material.forEach((m) => m.dispose());
        } else {
          line.material.dispose();
        }
      });
    };
  }, [scene]);

  // Update trajectory lines when trajectories change
  useEffect(() => {
    if (!refsRef.current || !scene) return;

    const { lines } = refsRef.current;

    Object.entries(trajectories).forEach(([name, trajectory]) => {
      const line = lines[name as OptimizerName];
      if (!line) return;

      const points: THREE.Vector3[] = [];
      trajectory.forEach((point) => {
        const worldCoords = paramsToWorldCoords(point.x, point.y, point.loss || 0);
        points.push(new THREE.Vector3(worldCoords.x, worldCoords.y, worldCoords.z));
      });

      line.geometry.setFromPoints(points);
      line.visible = showTrails && enabledOptimizers[name as OptimizerName] && points.length > 0;
    });
  }, [trajectories, showTrails, enabledOptimizers, scene, landscapeZRange]);

  // Update ball positions
  const updateBallPositions = (step: number) => {
    if (!refsRef.current) return;

    const { balls } = refsRef.current;

    Object.entries(trajectories).forEach(([name, trajectory]) => {
      const ball = balls[name as OptimizerName];
      if (!ball) return;

      const isEnabled = enabledOptimizers[name as OptimizerName];
      const hasTrajectory = trajectory && trajectory.length > 0;

      if (isEnabled && hasTrajectory && step < trajectory.length) {
        const point = trajectory[step];
        const worldCoords = paramsToWorldCoords(point.x, point.y, point.loss || point.z || 0);
        ball.position.set(worldCoords.x, worldCoords.y, worldCoords.z);
        ball.visible = true;
      } else {
        ball.visible = false;
      }
    });
  };

  // Convert parameter space to world coordinates
  const paramsToWorldCoords = (x: number, y: number, loss: number) => {
    const range = [-5, 5];
    const rangeMin = range[0];
    const rangeMax = range[1];
    const rangeSize = rangeMax - rangeMin;

    const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10;
    const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10;

    const worldY =
      ((loss - landscapeZRange.zMin) / (landscapeZRange.zMax - landscapeZRange.zMin)) *
      landscapeZRange.scale;

    return { x: worldX, y: worldY, z: worldZ };
  };

  return {
    updateBallPositions,
  };
};
