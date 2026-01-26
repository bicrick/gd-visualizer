import { create } from 'zustand'

export interface Manifold {
  id: string
  name: string
  description: string
  default_range: [number, number]
  parameters?: ManifoldParameter[]
}

export interface ManifoldParameter {
  name: string
  label: string
  min: number
  max: number
  step: number
  default: number
}

interface SceneState {
  // Manifold data
  manifolds: Manifold[]
  currentManifoldId: string
  manifoldParams: Record<string, number>
  manifoldRange: [number, number]
  
  // Starting position
  startX: number
  startY: number
  
  // Track last position used for optimization
  lastOptimizationStartPos: { x: number; y: number } | null
  
  // Trajectories (from optimization)
  trajectories: Record<string, Array<{ x: number; y: number; z: number }>>
  
  // Loading state
  isLoading: boolean
  loadingMessage: string
  
  // Computing state (specifically for optimization)
  isComputing: boolean
  
  // Actions
  setManifolds: (manifolds: Manifold[]) => void
  setCurrentManifold: (id: string) => void
  setManifoldParam: (name: string, value: number) => void
  setManifoldParams: (params: Record<string, number>) => void
  setStartPosition: (x: number, y: number) => void
  setTrajectories: (trajectories: Record<string, Array<{ x: number; y: number; z: number }>>) => void
  clearTrajectories: () => void
  setLoading: (isLoading: boolean, message?: string) => void
  setComputing: (isComputing: boolean) => void
  randomizeStartPosition: () => void
  setLastOptimizationPos: (x: number, y: number) => void
}

export const useSceneStore = create<SceneState>((set, get) => ({
  // Initial state
  manifolds: [],
  currentManifoldId: 'custom_multimodal',
  manifoldParams: {},
  manifoldRange: [-5, 5],
  startX: 3,
  startY: 3,
  lastOptimizationStartPos: null,
  trajectories: {},
  isLoading: false,
  loadingMessage: '',
  isComputing: false,
  
  // Actions
  setManifolds: (manifolds) => set({ manifolds }),
  
  setCurrentManifold: (id) => {
    const manifold = get().manifolds.find(m => m.id === id)
    if (manifold) {
      // Initialize default params for this manifold
      const params: Record<string, number> = {}
      manifold.parameters?.forEach(p => {
        params[p.name] = p.default
      })
      set({ 
        currentManifoldId: id, 
        manifoldParams: params,
        manifoldRange: manifold.default_range,
        trajectories: {}, // Clear trajectories on manifold change
        lastOptimizationStartPos: null,
        startX: 0, // Reset to center
        startY: 0,
      })
    }
  },
  
  setManifoldParam: (name, value) => set((state) => ({
    manifoldParams: { ...state.manifoldParams, [name]: value }
  })),
  
  setManifoldParams: (params) => set({ manifoldParams: params }),
  
  setStartPosition: (x, y) => {
    const { lastOptimizationStartPos } = get()
    // Clear trajectories if position differs significantly from last optimization
    const threshold = 0.01
    const shouldClear = lastOptimizationStartPos && (
      Math.abs(x - lastOptimizationStartPos.x) > threshold ||
      Math.abs(y - lastOptimizationStartPos.y) > threshold
    )
    
    if (shouldClear) {
      set({ startX: x, startY: y, trajectories: {}, lastOptimizationStartPos: null })
    } else {
      set({ startX: x, startY: y })
    }
  },
  
  setTrajectories: (trajectories) => set({ trajectories }),
  
  clearTrajectories: () => set({ trajectories: {}, lastOptimizationStartPos: null }),
  
  setLoading: (isLoading, message = '') => set({ isLoading, loadingMessage: message }),
  
  setComputing: (isComputing) => set({ isComputing }),
  
  randomizeStartPosition: () => {
    const [min, max] = get().manifoldRange
    const range = max - min
    set({
      startX: min + Math.random() * range,
      startY: min + Math.random() * range,
      trajectories: {}, // Clear old trajectories
      lastOptimizationStartPos: null,
    })
  },
  
  setLastOptimizationPos: (x, y) => set({ lastOptimizationStartPos: { x, y } }),
}))
