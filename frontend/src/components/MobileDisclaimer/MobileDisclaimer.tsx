import { useState, useEffect } from 'react'
import styles from './MobileDisclaimer.module.css'

const MOBILE_BREAKPOINT = 768

export const MobileDisclaimer = () => {
  const [isMobile, setIsMobile] = useState(false)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const checkViewport = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    }

    // Check on mount
    checkViewport()

    // Check on resize
    window.addEventListener('resize', checkViewport)
    return () => window.removeEventListener('resize', checkViewport)
  }, [])

  if (!isMobile || dismissed) {
    return null
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h1 className={styles.title}>Desktop Experience Recommended</h1>
        <p className={styles.message}>
          This website was not designed for mobile devices. 
          For the best experience, please open it on a desktop computer.
        </p>
        <button 
          className={styles.button}
          onClick={() => setDismissed(true)}
        >
          Proceed Anyway
        </button>
      </div>
    </div>
  )
}
