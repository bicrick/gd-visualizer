import { useState, useEffect } from 'react'
import { useUIStore } from '../../stores'
import styles from './GettingStartedModal.module.css'

const MOBILE_BREAKPOINT = 768

export const GettingStartedModal = () => {
  const gettingStartedModalOpen = useUIStore(state => state.gettingStartedModalOpen)
  const closeGettingStartedModal = useUIStore(state => state.closeGettingStartedModal)
  
  const [isMobile, setIsMobile] = useState(() => 
    typeof window !== 'undefined' && window.innerWidth < MOBILE_BREAKPOINT
  )

  useEffect(() => {
    const checkViewport = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    }
    window.addEventListener('resize', checkViewport)
    return () => window.removeEventListener('resize', checkViewport)
  }, [])

  if (!gettingStartedModalOpen) {
    return null
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h1 className={styles.title}>gd-visualizer</h1>
        
        <div className={styles.content}>
          {/* Animation Controls Section */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Animation Controls</h2>
            
            {isMobile ? (
              // Mobile animation controls
              <div className={styles.simpleControlGroup}>
                <div className={styles.simpleControlItem}>
                  <div className={`${styles.animationButtonWrapper} ${styles.playButtonWrapper}`}>
                    <svg className={styles.animationIcon} viewBox="0 0 24 24" fill="none">
                      <path d="M8 5v14l11-7z" fill="currentColor"/>
                    </svg>
                  </div>
                  <span className={styles.simpleDescription}>Start/pause animation</span>
                </div>
                
                <div className={styles.simpleControlItem}>
                  <div className={`${styles.animationButtonWrapper} ${styles.pickButtonWrapper}`}>
                    <svg className={styles.animationIcon} viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="2" fill="currentColor"/>
                      <circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" strokeWidth="2"/>
                      <line x1="12" y1="2" x2="12" y2="7" stroke="currentColor" strokeWidth="2"/>
                      <line x1="12" y1="17" x2="12" y2="22" stroke="currentColor" strokeWidth="2"/>
                      <line x1="2" y1="12" x2="7" y2="12" stroke="currentColor" strokeWidth="2"/>
                      <line x1="17" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </div>
                  <span className={styles.simpleDescription}>Pick starting point</span>
                </div>
              </div>
            ) : (
              // Desktop animation controls
              <div className={styles.controlGroup}>
                <div className={styles.controlItem}>
                  <div className={`${styles.animationButtonWrapper} ${styles.playButtonWrapper}`}>
                    <svg className={styles.animationIcon} viewBox="0 0 24 24" fill="none">
                      <path d="M8 5v14l11-7z" fill="currentColor"/>
                    </svg>
                  </div>
                  <span className={styles.description}>Start/pause animation</span>
                </div>
                
                <div className={styles.controlItem}>
                  <div className={`${styles.animationButtonWrapper} ${styles.pickButtonWrapper}`}>
                    <svg className={styles.animationIcon} viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="2" fill="currentColor"/>
                      <circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" strokeWidth="2"/>
                      <line x1="12" y1="2" x2="12" y2="7" stroke="currentColor" strokeWidth="2"/>
                      <line x1="12" y1="17" x2="12" y2="22" stroke="currentColor" strokeWidth="2"/>
                      <line x1="2" y1="12" x2="7" y2="12" stroke="currentColor" strokeWidth="2"/>
                      <line x1="17" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </div>
                  <span className={styles.description}>Pick starting point</span>
                </div>
              </div>
            )}
          </div>

          {/* Camera Controls Section */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Camera Controls</h2>
            
            {isMobile ? (
              // Mobile touch controls
              <div className={styles.simpleControlGroup}>
                <div className={styles.simpleControlItem}>
                  <span className={styles.gesture}>One Finger</span>
                  <span className={styles.simpleDescription}>Rotate camera</span>
                </div>
                
                <div className={styles.simpleControlItem}>
                  <span className={styles.gesture}>Pinch</span>
                  <span className={styles.simpleDescription}>Zoom in/out</span>
                </div>
                
                <div className={styles.simpleControlItem}>
                  <span className={styles.gesture}>Two Fingers</span>
                  <span className={styles.simpleDescription}>Pan camera</span>
                </div>
              </div>
            ) : (
              // Desktop keyboard/mouse controls
              <div className={styles.controlGroup}>
                <div className={styles.controlItem}>
                  <div className={styles.keyboard}>
                    <div className={styles.keyRow}>
                      <div className={styles.key}>W</div>
                    </div>
                    <div className={styles.keyRow}>
                      <div className={styles.key}>A</div>
                      <div className={styles.key}>S</div>
                      <div className={styles.key}>D</div>
                    </div>
                  </div>
                  <span className={styles.description}>Navigate landscape horizontally</span>
                </div>
                
                <div className={styles.controlItem}>
                  <div className={styles.mouseControl}>
                    <svg className={styles.mouseIcon} viewBox="0 0 24 24" fill="none">
                      <rect x="7" y="4" width="10" height="16" rx="4" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                    <span className={styles.mouseButton}>L</span>
                  </div>
                  <span className={styles.description}>Rotate camera</span>
                </div>
                
                <div className={styles.controlItem}>
                  <div className={styles.mouseControl}>
                    <svg className={styles.mouseIcon} viewBox="0 0 24 24" fill="none">
                      <rect x="7" y="4" width="10" height="16" rx="4" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                    <span className={styles.mouseButton}>R</span>
                  </div>
                  <span className={styles.description}>Pan camera</span>
                </div>
              </div>
            )}
          </div>

          {/* Tab Bar Section (Mobile only) */}
          {isMobile && (
            <div className={styles.section}>
              <h2 className={styles.sectionTitle}>Navigation</h2>
              <div className={styles.simpleControlGroup}>
                <p className={styles.simpleDescription}>
                  Use the bottom tabs to access manifold selection, parameters, and optimizer settings.
                  The play button starts the animation.
                </p>
              </div>
            </div>
          )}
        </div>
        
        <button 
          className={styles.button}
          onClick={closeGettingStartedModal}
        >
          Get Started
        </button>
      </div>
    </div>
  )
}
