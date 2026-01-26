import { create } from 'zustand'

type Theme = 'dark' | 'light'

interface UIState {
  // Theme
  theme: Theme
  
  // Modal state
  settingsModalOpen: boolean
  
  // Picking mode for start position
  pickingMode: boolean
  
  // Actions
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  openSettingsModal: () => void
  closeSettingsModal: () => void
  setPickingMode: (enabled: boolean) => void
}

export const useUIStore = create<UIState>((set, get) => ({
  // Initial state - read from localStorage
  theme: (localStorage.getItem('theme') as Theme) || 'dark',
  settingsModalOpen: false,
  pickingMode: false,
  
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
  setPickingMode: (enabled) => set({ pickingMode: enabled }),
}))
