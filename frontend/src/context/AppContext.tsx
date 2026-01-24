import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import {
  AppContextValue,
  AppState,
  OptimizerName,
  OptimizerParams,
  AnimationState,
  Theme,
  ManifoldInfo,
  ManifoldParams,
} from './types';

// Default optimizer parameters
const defaultOptimizerParams: OptimizerParams = {
  sgd: {
    learningRate: 0.01,
    useConvergence: true,
    iterations: 100,
    maxIterations: 10000,
    convergenceThreshold: 0.0001,
  },
  batch: {
    learningRate: 0.01,
    useConvergence: true,
    iterations: 100,
    maxIterations: 10000,
    convergenceThreshold: 0.0001,
  },
  momentum: {
    learningRate: 0.01,
    momentum: 0.9,
    lrDecay: 0.995,
    useConvergence: true,
    iterations: 100,
    maxIterations: 10000,
    convergenceThreshold: 0.0001,
  },
  adam: {
    learningRate: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 0.00000001,
    useConvergence: true,
    iterations: 100,
    maxIterations: 10000,
    convergenceThreshold: 0.0001,
  },
  ballistic: {
    dropHeight: 5.0,
    gravity: 1.0,
    elasticity: 0.8,
    bounceThreshold: 0.05,
    ballRadius: 0.05,
    maxIterations: 10000,
  },
  ballistic_adam: {
    learningRate: 0.01,
    momentum: 0.9,
    gravity: 0.001,
    dt: 1.0,
    maxAirSteps: 20,
    maxBisectionIters: 10,
    collisionTol: 0.001,
    useConvergence: true,
    iterations: 100,
    maxIterations: 10000,
    convergenceThreshold: 0.0001,
  },
};

// Default state
const defaultState: AppState = {
  startPosition: { x: 3, y: 3 },
  animationState: 'stopped',
  currentStep: 0,
  totalSteps: 0,
  speed: 1.0,
  showTrails: true,
  pickingMode: false,
  optimizerParams: defaultOptimizerParams,
  enabledOptimizers: {
    sgd: false,
    batch: true,
    momentum: true,
    adam: true,
    ballistic: true,
    ballistic_adam: true,
  },
  trajectories: {},
  currentManifoldId: '',
  manifoldParams: {},
  manifolds: [],
  theme: (localStorage.getItem('theme') as Theme) || 'dark',
};

const AppContext = createContext<AppContextValue | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AppState>(defaultState);

  // Start position actions
  const setStartPosition = useCallback((x: number, y: number) => {
    setState((prev) => ({
      ...prev,
      startPosition: { x, y },
    }));
  }, []);

  const randomizeStartPosition = useCallback(() => {
    const x = Math.random() * 10 - 5; // Random between -5 and 5
    const y = Math.random() * 10 - 5;
    setStartPosition(x, y);
  }, [setStartPosition]);

  // Animation actions
  const setAnimationState = useCallback((animationState: AnimationState) => {
    setState((prev) => ({
      ...prev,
      animationState,
    }));
  }, []);

  const setCurrentStep = useCallback((currentStep: number) => {
    setState((prev) => ({
      ...prev,
      currentStep,
    }));
  }, []);

  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({
      ...prev,
      speed,
    }));
  }, []);

  // Display actions
  const setShowTrails = useCallback((showTrails: boolean) => {
    setState((prev) => ({
      ...prev,
      showTrails,
    }));
  }, []);

  const setPickingMode = useCallback((pickingMode: boolean) => {
    setState((prev) => ({
      ...prev,
      pickingMode,
    }));
  }, []);

  // Optimizer actions
  const updateOptimizerParam = useCallback(
    <T extends OptimizerName>(
      optimizer: T,
      param: keyof OptimizerParams[T],
      value: number | boolean
    ) => {
      setState((prev) => ({
        ...prev,
        optimizerParams: {
          ...prev.optimizerParams,
          [optimizer]: {
            ...prev.optimizerParams[optimizer],
            [param]: value,
          },
        },
      }));
    },
    []
  );

  const toggleOptimizer = useCallback((optimizer: OptimizerName, enabled: boolean) => {
    setState((prev) => ({
      ...prev,
      enabledOptimizers: {
        ...prev.enabledOptimizers,
        [optimizer]: enabled,
      },
    }));
  }, []);

  // Manifold actions
  const setManifold = useCallback((manifoldId: string) => {
    setState((prev) => {
      const manifold = prev.manifolds.find((m) => m.id === manifoldId);
      const newManifoldParams: ManifoldParams = {};
      
      if (manifold?.params) {
        manifold.params.forEach((param) => {
          newManifoldParams[param.id] = param.default;
        });
      }

      return {
        ...prev,
        currentManifoldId: manifoldId,
        manifoldParams: newManifoldParams,
        trajectories: {}, // Clear trajectories when changing manifold
        animationState: 'stopped',
        currentStep: 0,
        totalSteps: 0,
      };
    });
  }, []);

  const updateManifoldParam = useCallback((param: string, value: number) => {
    setState((prev) => ({
      ...prev,
      manifoldParams: {
        ...prev.manifoldParams,
        [param]: value,
      },
    }));
  }, []);

  const setManifolds = useCallback((manifolds: ManifoldInfo[]) => {
    setState((prev) => ({
      ...prev,
      manifolds,
    }));
  }, []);

  // Optimization actions
  const runOptimization = useCallback(async () => {
    try {
      const enabledOpts: any = {};
      
      (Object.keys(state.enabledOptimizers) as OptimizerName[]).forEach((name) => {
        if (state.enabledOptimizers[name]) {
          enabledOpts[name] = {
            ...state.optimizerParams[name],
            enabled: true,
          };
        }
      });

      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          startX: state.startPosition.x,
          startY: state.startPosition.y,
          manifoldId: state.currentManifoldId,
          manifoldParams: state.manifoldParams,
          optimizers: enabledOpts,
        }),
      });

      if (!response.ok) {
        throw new Error('Optimization request failed');
      }

      const data = await response.json();
      
      // Calculate total steps from trajectories
      const trajectoryLengths = Object.values(data.results).map(
        (traj: any) => traj.length
      );
      const maxSteps = Math.max(...trajectoryLengths, 0);

      setState((prev) => ({
        ...prev,
        trajectories: data.results,
        totalSteps: maxSteps,
        currentStep: 0,
        animationState: 'stopped',
      }));
    } catch (error) {
      console.error('Optimization failed:', error);
    }
  }, [state]);

  const clearTrajectories = useCallback(() => {
    setState((prev) => ({
      ...prev,
      trajectories: {},
      currentStep: 0,
      totalSteps: 0,
      animationState: 'stopped',
    }));
  }, []);

  // Theme actions
  const setTheme = useCallback((theme: Theme) => {
    localStorage.setItem('theme', theme);
    document.body.setAttribute('data-theme', theme);
    setState((prev) => ({
      ...prev,
      theme,
    }));
  }, []);

  const toggleTheme = useCallback(() => {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  }, [state.theme, setTheme]);

  const value: AppContextValue = {
    ...state,
    setStartPosition,
    randomizeStartPosition,
    setAnimationState,
    setCurrentStep,
    setSpeed,
    setShowTrails,
    setPickingMode,
    updateOptimizerParam,
    toggleOptimizer,
    setManifold,
    updateManifoldParam,
    setManifolds,
    runOptimization,
    clearTrajectories,
    setTheme,
    toggleTheme,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = (): AppContextValue => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};
