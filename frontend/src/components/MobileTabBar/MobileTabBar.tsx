import { useUIStore, MobilePanel } from '../../stores/uiStore'
import styles from './MobileTabBar.module.css'

interface Tab {
  id: MobilePanel
  label: string
  icon: React.ReactNode
}

const tabs: Tab[] = [
  {
    id: 'manifold',
    label: 'Manifold',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: 'params',
    label: 'Params',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="4" y1="21" x2="4" y2="14" />
        <line x1="4" y1="10" x2="4" y2="3" />
        <line x1="12" y1="21" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12" y2="3" />
        <line x1="20" y1="21" x2="20" y2="16" />
        <line x1="20" y1="12" x2="20" y2="3" />
        <circle cx="4" cy="12" r="2" />
        <circle cx="12" cy="10" r="2" />
        <circle cx="20" cy="14" r="2" />
      </svg>
    ),
  },
  {
    id: 'optimizers',
    label: 'Optimizers',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
  },
]

export function MobileTabBar() {
  const activeMobilePanel = useUIStore(state => state.activeMobilePanel)
  const setActiveMobilePanel = useUIStore(state => state.setActiveMobilePanel)

  const handleTabClick = (tabId: MobilePanel) => {
    if (activeMobilePanel === tabId) {
      setActiveMobilePanel(null)
    } else {
      setActiveMobilePanel(tabId)
    }
  }

  return (
    <div className={styles.tabBar}>
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`${styles.tab} ${activeMobilePanel === tab.id ? styles.active : ''}`}
          onClick={() => handleTabClick(tab.id)}
        >
          <span className={styles.icon}>{tab.icon}</span>
          <span className={styles.label}>{tab.label}</span>
        </button>
      ))}
    </div>
  )
}
