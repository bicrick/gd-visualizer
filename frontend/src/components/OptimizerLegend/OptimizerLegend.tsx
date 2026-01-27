import { useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import styles from './OptimizerLegend.module.css'

export function OptimizerLegend() {
  const enabled = useOptimizerStore(state => state.enabled)
  
  const enabledOptimizers = Object.entries(enabled)
    .filter(([_, isEnabled]) => isEnabled)
    .map(([name]) => ({
      name,
      color: OPTIMIZER_COLORS[name] || '#ffffff',
      displayName: name.charAt(0).toUpperCase() + name.slice(1)
    }))
  
  if (enabledOptimizers.length === 0) {
    return null
  }
  
  return (
    <div className={styles.legend}>
      {enabledOptimizers.map(({ name, color, displayName }) => (
        <div key={name} className={styles.item}>
          <div 
            className={styles.colorDot} 
            style={{ backgroundColor: color }}
          />
          <span className={styles.name}>{displayName}</span>
        </div>
      ))}
    </div>
  )
}
