import { useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import { useUIStore } from '../../stores/uiStore'
import styles from './MobileHeader.module.css'

export function MobileHeader() {
  const enabled = useOptimizerStore(state => state.enabled)
  const theme = useUIStore(state => state.theme)
  const toggleTheme = useUIStore(state => state.toggleTheme)
  const openSettingsModal = useUIStore(state => state.openSettingsModal)

  const enabledOptimizers = Object.entries(enabled)
    .filter(([_, isEnabled]) => isEnabled)
    .map(([name]) => ({
      name,
      color: OPTIMIZER_COLORS[name] || '#ffffff',
      displayName: name.charAt(0).toUpperCase() + name.slice(1)
    }))

  return (
    <div className={styles.header}>
      <div className={styles.legend}>
        {enabledOptimizers.map(({ name, color, displayName }) => (
          <div key={name} className={styles.legendItem}>
            <span className={styles.dot} style={{ backgroundColor: color }} />
            <span className={styles.name}>{displayName}</span>
          </div>
        ))}
      </div>
      
      <div className={styles.actions}>
        <button 
          className={styles.iconBtn}
          onClick={openSettingsModal}
          aria-label="Settings"
        >
          <svg viewBox="0 0 24 24" className={styles.settingsIcon}>
            <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
        </button>
        
        <button 
          className={styles.iconBtn}
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <circle cx="12" cy="12" r="5" />
              <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
