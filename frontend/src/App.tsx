import { useState, useEffect } from 'react'
import { Canvas3D } from './components/Canvas3D'
import { ManifoldCard } from './components/cards/ManifoldCard'
import { ManifoldParamsCard } from './components/cards/ManifoldParamsCard'
import { AnimationCard } from './components/cards/AnimationCard'
import { OptimizersCard } from './components/cards/OptimizersCard'
import { OptimizerLegend } from './components/OptimizerLegend'
import { SettingsButton } from './components/SettingsButton'
import { ThemeToggle } from './components/ThemeToggle'
import { SettingsModal } from './components/SettingsModal'
import { GettingStartedModal } from './components/GettingStartedModal'
import { MobileHeader } from './components/MobileHeader'
import { MobileTabBar } from './components/MobileTabBar'
import { MobilePanels } from './components/MobilePanels'
import { MobileFloatingControls } from './components/MobileFloatingControls'
import styles from './App.module.css'

const MOBILE_BREAKPOINT = 768

function useMobileDetection() {
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

  return isMobile
}

function App() {
  const isMobile = useMobileDetection()

  return (
    <div className={styles.container}>
      {/* Getting started modal */}
      <GettingStartedModal />
      
      {/* Full-screen 3D canvas */}
      <Canvas3D isMobile={isMobile} />
      
      {isMobile ? (
        <>
          {/* Mobile Layout */}
          <MobileHeader />
          <MobileFloatingControls />
          <MobilePanels />
          <MobileTabBar />
        </>
      ) : (
        <>
          {/* Desktop Layout */}
          <div className={styles.topLeft}>
            <ManifoldCard />
          </div>
          
          <OptimizerLegend />
          
          <div className={styles.topRight}>
            <ManifoldParamsCard />
          </div>
          
          <div className={styles.bottomLeft}>
            <AnimationCard />
          </div>
          
          <div className={styles.bottomRight}>
            <OptimizersCard />
          </div>
          
          {/* Floating buttons */}
          <div className={styles.floatingButtons}>
            <SettingsButton />
            <ThemeToggle />
          </div>
        </>
      )}
      
      {/* Modals - available on both mobile and desktop */}
      <SettingsModal />
    </div>
  )
}

export default App
