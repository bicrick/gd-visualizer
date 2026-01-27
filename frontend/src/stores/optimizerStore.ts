import { create } from 'zustand'
import { useAnimationStore } from './animationStore'

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

export interface SgdParams extends OptimizerParams {
  stepMultiplier: number  // Multiplier for effective step size (faster convergence)
  noiseScale: number      // Magnitude of gradient noise (bouncing behavior)
  noiseDecay: number      // Decay factor per iteration (allows settling over time)
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
  sgd: SgdParams
  
  // Panel state
  activePanelOptimizer: string | null
  
  // Actions
  toggleOptimizer: (name: keyof OptimizerState['enabled'], enabled: boolean) => void
  setOptimizerParam: <T extends keyof OptimizerState['enabled']>(
    optimizer: T, 
    param: string, 
    value: number | boolean
  ) => void
  openOptimizerPanel: (name: string) => void
  closeOptimizerPanel: () => void
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
    stepMultiplier: 3.0,  // SGD takes 3x larger steps (faster convergence)
    noiseScale: 0.8,      // High noise for visible "bouncing"
    noiseDecay: 0.995,    // Noise reduces to ~60% at 100 steps, ~8% at 500 steps
  },
  
  // Initial panel state
  activePanelOptimizer: null,
  
  // Actions
  toggleOptimizer: (name, enabled) => {
    // Stop animation when optimizer enabled state changes
    useAnimationStore.getState().stop()
    set((state) => ({
      enabled: { ...state.enabled, [name]: enabled }
    }))
  },
  
  setOptimizerParam: (optimizer, param, value) => {
    // Stop animation when optimizer parameters change
    useAnimationStore.getState().stop()
    set((state) => ({
      [optimizer]: { ...state[optimizer], [param]: value }
    }))
  },
  
  openOptimizerPanel: (name) => set({ activePanelOptimizer: name }),
  
  closeOptimizerPanel: () => set({ activePanelOptimizer: null }),
  
  getEnabledOptimizers: () => get().enabled,
  
  getOptimizerParams: () => ({
    batch: get().batch,
    momentum: get().momentum,
    adam: get().adam,
    sgd: get().sgd,
  }),
}))
