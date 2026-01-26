import { useEffect, useCallback } from 'react'
import { useUIStore, useAnimationStore } from '../../stores'
import styles from './SettingsModal.module.css'

export function SettingsModal() {
  const isOpen = useUIStore(state => state.settingsModalOpen)
  const closeSettingsModal = useUIStore(state => state.closeSettingsModal)
  
  const showTrails = useAnimationStore(state => state.showTrails)
  const setShowTrails = useAnimationStore(state => state.setShowTrails)
  
  // Close on escape key
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      closeSettingsModal()
    }
  }, [closeSettingsModal])
  
  useEffect(() => {
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, handleKeyDown])
  
  if (!isOpen) return null
  
  return (
    <div className={styles.overlay} onClick={closeSettingsModal}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>Settings</h2>
          <button className={styles.closeBtn} onClick={closeSettingsModal}>
            <svg viewBox="0 0 24 24" className={styles.closeIcon}>
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className={styles.content}>
          {/* Display Options Section */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Display Options</h3>
            
            <div className={styles.settingItem}>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={showTrails}
                  onChange={(e) => setShowTrails(e.target.checked)}
                />
                <span className={styles.checkmark} />
                <span className={styles.labelText}>Show Trajectory Trails</span>
              </label>
              <p className={styles.settingDescription}>
                Display the optimization path as a colored line following each optimizer's trajectory.
              </p>
            </div>
          </div>
          
          {/* Placeholder for future settings */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>More Settings Coming Soon</h3>
            <p className={styles.placeholder}>
              Additional settings and customization options will be added here.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
