import { create } from 'zustand'

export interface OptimizerParams {
  learningRate: number
  iterations: number
  useConvergence: boolean
  maxIterations: number
  convergenceThreshold: number
}

export interface MomentumParams extends OptimizerParams {
  momentum: number
  lrDecay: number
}

export interface AdamParams extends OptimizerParams {
  beta1: number
  beta2: number
  epsilon: number
}

interface OptimizerState {
  // Enabled state for each optimizer
  enabled: {
    batch: boolean
    momentum: boolean
    adam: boolean
    sgd: boolean
  }
  
  // Parameters for each optimizer
  batch: OptimizerParams
  momentum: MomentumParams
  adam: AdamParams
  sgd: OptimizerParams
  
  // Actions
  toggleOptimizer: (name: keyof OptimizerState['enabled'], enabled: boolean) => void
  setOptimizerParam: <T extends keyof OptimizerState['enabled']>(
    optimizer: T, 
    param: string, 
    value: number | boolean
  ) => void
  getEnabledOptimizers: () => Record<string, boolean>
  getOptimizerParams: () => Record<string, unknown>
}

// Optimizer colors for visualization
export const OPTIMIZER_COLORS: Record<string, string> = {
  batch: '#4444ff',
  momentum: '#44ff44',
  adam: '#ff8800',
  sgd: '#ff4444',
}

export const useOptimizerStore = create<OptimizerState>((set, get) => ({
  // Initial enabled state
  enabled: {
    batch: true,
    momentum: true,
    adam: true,
    sgd: false,
  },
  
  // Initial parameters for each optimizer
  batch: {
    learningRate: 0.01,
    iterations: 100,
    useConvergence: true,
    maxIterations: 10000,
    convergenceThreshold: 1e-4,
  },
  
  momentum: {
    learningRate: 0.01,
    momentum: 0.9,
    lrDecay: 0.995,
    iterations: 100,
    useConvergence: true,
    maxIterations: 10000,
    convergenceThreshold: 1e-4,
  },
  
  adam: {
    learningRate: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    iterations: 100,
    useConvergence: true,
    maxIterations: 10000,
    convergenceThreshold: 1e-4,
  },
  
  sgd: {
    learningRate: 0.01,
    iterations: 100,
    useConvergence: true,
    maxIterations: 10000,
    convergenceThreshold: 1e-4,
  },
  
  // Actions
  toggleOptimizer: (name, enabled) => set((state) => ({
    enabled: { ...state.enabled, [name]: enabled }
  })),
  
  setOptimizerParam: (optimizer, param, value) => set((state) => ({
    [optimizer]: { ...state[optimizer], [param]: value }
  })),
  
  getEnabledOptimizers: () => get().enabled,
  
  getOptimizerParams: () => ({
    batch: get().batch,
    momentum: get().momentum,
    adam: get().adam,
    sgd: get().sgd,
  }),
}))
