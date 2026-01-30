import { create } from 'zustand'

type Theme = 'dark' | 'light'
export type MobilePanel = 'manifold' | 'params' | 'optimizers' | null

interface UIState {
  // Theme
  theme: Theme
  
  // Modal state
  settingsModalOpen: boolean
  gettingStartedModalOpen: boolean
  
  // Picking mode for start position
  pickingMode: boolean
  
  // Mobile state
  activeMobilePanel: MobilePanel
  
  // Actions
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  openSettingsModal: () => void
  closeSettingsModal: () => void
  closeGettingStartedModal: () => void
  setPickingMode: (enabled: boolean) => void
  setActiveMobilePanel: (panel: MobilePanel) => void
}

export const useUIStore = create<UIState>((set, get) => ({
  // Initial state - read from localStorage
  theme: (localStorage.getItem('theme') as Theme) || 'dark',
  settingsModalOpen: false,
  gettingStartedModalOpen: true,
  pickingMode: false,
  activeMobilePanel: null,
  
  // Actions
  setTheme: (theme) => {
    localStorage.setItem('theme', theme)
    document.documentElement.setAttribute('data-theme', theme)
    set({ theme })
  },
  
  toggleTheme: () => {
    const newTheme = get().theme === 'dark' ? 'light' : 'dark'
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    set({ theme: newTheme })
  },
  
  openSettingsModal: () => set({ settingsModalOpen: true }),
  closeSettingsModal: () => set({ settingsModalOpen: false }),
  closeGettingStartedModal: () => set({ gettingStartedModalOpen: false }),
  setPickingMode: (enabled) => set({ pickingMode: enabled }),
  setActiveMobilePanel: (panel) => set({ activeMobilePanel: panel }),
}))
