import { Canvas3D } from './components/Canvas3D'
import { ManifoldCard } from './components/cards/ManifoldCard'
import { ManifoldParamsCard } from './components/cards/ManifoldParamsCard'
import { AnimationCard } from './components/cards/AnimationCard'
import { OptimizersCard } from './components/cards/OptimizersCard'
import { SettingsButton } from './components/SettingsButton'
import { ThemeToggle } from './components/ThemeToggle'
import { SettingsModal } from './components/SettingsModal'
import { MobileDisclaimer } from './components/MobileDisclaimer'
import styles from './App.module.css'

function App() {
  return (
    <div className={styles.container}>
      {/* Mobile disclaimer overlay */}
      <MobileDisclaimer />
      
      {/* Full-screen 3D canvas */}
      <Canvas3D />
      
      {/* Floating cards */}
      <div className={styles.topLeft}>
        <ManifoldCard />
      </div>
      
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
      
      {/* Modals */}
      <SettingsModal />
    </div>
  )
}

export default App
