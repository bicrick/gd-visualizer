import { Card } from '../Card'
import { useSceneStore } from '../../stores'
import styles from './ManifoldParamsCard.module.css'

export function ManifoldParamsCard() {
  const manifolds = useSceneStore(state => state.manifolds)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const manifoldParams = useSceneStore(state => state.manifoldParams)
  const setManifoldParam = useSceneStore(state => state.setManifoldParam)
  
  const currentManifold = manifolds.find(m => m.id === currentManifoldId)
  const parameters = currentManifold?.parameters || []
  
  // Don't render if no parameters
  if (parameters.length === 0) {
    return null
  }
  
  return (
    <Card 
      title="Parameters" 
      summary={`${parameters.length} params`}
      defaultCollapsed={true}
    >
      <div className={styles.params}>
        {parameters.map(param => {
          const value = manifoldParams[param.name] ?? param.default
          const isInteger = param.step >= 1
          
          return (
            <div key={param.name} className={styles.paramControl}>
              <div className={styles.paramLabel}>
                <span className={styles.paramName}>{param.label}</span>
                <span className={styles.paramValue}>
                  {isInteger ? value.toString() : value.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                className={styles.paramSlider}
                min={param.min}
                max={param.max}
                step={param.step}
                value={value}
                onChange={(e) => setManifoldParam(param.name, parseFloat(e.target.value))}
              />
            </div>
          )
        })}
      </div>
    </Card>
  )
}
