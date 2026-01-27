import { useUIStore } from '../../stores'
import styles from './GettingStartedModal.module.css'

export const GettingStartedModal = () => {
  const gettingStartedModalOpen = useUIStore(state => state.gettingStartedModalOpen)
  const closeGettingStartedModal = useUIStore(state => state.closeGettingStartedModal)

  if (!gettingStartedModalOpen) {
    return null
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h1 className={styles.title}>gd-visualizer</h1>
        
        <div className={styles.content}>
          {/* Camera Controls Section */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Camera Controls</h2>
            
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
          </div>

          {/* Action Buttons Section */}
          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Action Controls</h2>
            
            <div className={styles.controlGroup}>
              <div className={styles.controlItem}>
                <div className={`${styles.actionButton} ${styles.playButton}`}>
                  <svg className={styles.buttonIcon} viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                </div>
                <span className={styles.description}>Start optimization animation</span>
              </div>
              
              <div className={styles.controlItem}>
                <div className={`${styles.actionButton} ${styles.pickButton}`}>
                  <svg className={styles.buttonIcon} viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="2"/>
                    <circle cx="12" cy="12" r="7" fill="none" stroke="white" strokeWidth="2"/>
                    <line x1="12" y1="2" x2="12" y2="7" stroke="white" strokeWidth="2"/>
                    <line x1="12" y1="17" x2="12" y2="22" stroke="white" strokeWidth="2"/>
                    <line x1="2" y1="12" x2="7" y2="12" stroke="white" strokeWidth="2"/>
                    <line x1="17" y1="12" x2="22" y2="12" stroke="white" strokeWidth="2"/>
                  </svg>
                </div>
                <span className={styles.description}>Select starting position on landscape</span>
              </div>
            </div>
          </div>
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
