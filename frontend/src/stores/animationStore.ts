import { create } from 'zustand'

type AnimationState = 'stopped' | 'playing' | 'paused'

interface AnimationStoreState {
  // Animation state
  state: AnimationState
  currentStep: number
  totalSteps: number
  speed: number
  
  // Display options
  showTrails: boolean
  
  // Actions
  play: () => void
  pause: () => void
  stop: () => void
  setCurrentStep: (step: number) => void
  setTotalSteps: (steps: number) => void
  setSpeed: (speed: number) => void
  setShowTrails: (show: boolean) => void
  incrementStep: () => void
}

export const useAnimationStore = create<AnimationStoreState>((set, get) => ({
  // Initial state
  state: 'stopped',
  currentStep: 0,
  totalSteps: 0,
  speed: 1.0,
  showTrails: true,
  
  // Actions
  play: () => set({ state: 'playing' }),
  pause: () => set({ state: 'paused' }),
  stop: () => set({ state: 'stopped' }),
  
  setCurrentStep: (step) => set({ currentStep: step }),
  setTotalSteps: (steps) => set({ totalSteps: steps }),
  setSpeed: (speed) => set({ speed }),
  setShowTrails: (show) => set({ showTrails: show }),
  
  incrementStep: () => {
    const { currentStep, totalSteps } = get()
    if (currentStep < totalSteps) {
      set({ currentStep: currentStep + 1 })
    } else {
      set({ state: 'paused' })
    }
  },
}))
