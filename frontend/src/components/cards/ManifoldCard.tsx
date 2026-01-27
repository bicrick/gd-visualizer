import { useEffect } from 'react'
import { Card } from '../Card'
import { useSceneStore } from '../../stores'
import { fetchManifolds } from '../../utils/api'
import styles from './ManifoldCard.module.css'

export function ManifoldCard() {
  const manifolds = useSceneStore(state => state.manifolds)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const setManifolds = useSceneStore(state => state.setManifolds)
  const setCurrentManifold = useSceneStore(state => state.setCurrentManifold)
  
  // Fetch manifolds on mount and initialize params
  useEffect(() => {
    fetchManifolds()
      .then(data => {
        setManifolds(data.manifolds)
        // Find the current manifold and initialize its params
        const currentManifold = data.manifolds.find(m => m.id === currentManifoldId)
        if (currentManifold) {
          setCurrentManifold(currentManifoldId)
        }
      })
      .catch(err => {
        console.error('Failed to load manifolds:', err)
      })
  }, []) // Only run once on mount
  
  const currentManifold = manifolds.find(m => m.id === currentManifoldId)
  const currentName = currentManifold?.name || 'Loading...'
  
  const handleSelect = (id: string) => {
    setCurrentManifold(id)
  }
  
  return (
    <Card 
      title="Manifold" 
      summary={currentName}
      defaultCollapsed={true}
    >
      <div className={styles.options}>
        {manifolds.map(manifold => (
          <div 
            key={manifold.id}
            className={`${styles.option} ${manifold.id === currentManifoldId ? styles.selected : ''}`}
            onClick={() => handleSelect(manifold.id)}
          >
            <div className={styles.optionName}>{manifold.name}</div>
            <div className={styles.optionDescription}>{manifold.description}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}
