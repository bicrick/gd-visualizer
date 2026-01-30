import { useEffect, useRef, ReactNode } from 'react'
import { useUIStore } from '../../stores/uiStore'
import styles from './MobileBottomSheet.module.css'

interface MobileBottomSheetProps {
  children: ReactNode
  title: string
}

export function MobileBottomSheet({ children, title }: MobileBottomSheetProps) {
  const activeMobilePanel = useUIStore(state => state.activeMobilePanel)
  const setActiveMobilePanel = useUIStore(state => state.setActiveMobilePanel)
  const sheetRef = useRef<HTMLDivElement>(null)

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && activeMobilePanel) {
        setActiveMobilePanel(null)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [activeMobilePanel, setActiveMobilePanel])

  // Close when clicking backdrop
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      setActiveMobilePanel(null)
    }
  }

  if (!activeMobilePanel) return null

  return (
    <div className={styles.backdrop} onClick={handleBackdropClick}>
      <div ref={sheetRef} className={styles.sheet}>
        <div className={styles.handle} />
        <div className={styles.header}>
          <h2 className={styles.title}>{title}</h2>
          <button 
            className={styles.closeBtn}
            onClick={() => setActiveMobilePanel(null)}
            aria-label="Close"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className={styles.content}>
          {children}
        </div>
      </div>
    </div>
  )
}
