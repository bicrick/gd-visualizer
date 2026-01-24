import { useEffect, useRef } from 'react';
import { useAppContext } from '../context/AppContext';

export const useAnimation = (updateBallPositions: (step: number) => void) => {
  const { animationState, currentStep, setCurrentStep, speed, totalSteps, trajectories } =
    useAppContext();
  const animationFrameRef = useRef<number | null>(null);
  const lastUpdateTimeRef = useRef<number>(0);

  useEffect(() => {
    if (animationState !== 'playing') {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const animate = (timestamp: number) => {
      if (!lastUpdateTimeRef.current) {
        lastUpdateTimeRef.current = timestamp;
      }

      const elapsed = timestamp - lastUpdateTimeRef.current;
      const interval = 1000 / (30 * speed); // 30 fps base rate * speed

      if (elapsed >= interval) {
        const nextStep = currentStep + 1;

        if (nextStep >= totalSteps) {
          setCurrentStep(totalSteps - 1);
          updateBallPositions(totalSteps - 1);
          return;
        }

        setCurrentStep(nextStep);
        updateBallPositions(nextStep);
        lastUpdateTimeRef.current = timestamp;
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [animationState, currentStep, speed, totalSteps, updateBallPositions, setCurrentStep]);

  // Update ball positions when currentStep changes (from scrubbing)
  useEffect(() => {
    if (Object.keys(trajectories).length > 0) {
      updateBallPositions(currentStep);
    }
  }, [currentStep, trajectories, updateBallPositions]);
};
