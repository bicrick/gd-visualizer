import { useCallback } from 'react'
import { useSceneStore, useAnimationStore, useOptimizerStore, useUIStore } from '../../stores'
import { runOptimization } from '../../utils/api'
import styles from './MobileFloatingControls.module.css'

export function MobileFloatingControls() {
  const startX = useSceneStore(state => state.startX)
  const startY = useSceneStore(state => state.startY)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const manifoldParams = useSceneStore(state => state.manifoldParams)
  const setTrajectories = useSceneStore(state => state.setTrajectories)
  const setLoading = useSceneStore(state => state.setLoading)
  const isComputing = useSceneStore(state => state.isComputing)
  const setComputing = useSceneStore(state => state.setComputing)
  const lastOptimizationStartPos = useSceneStore(state => state.lastOptimizationStartPos)
  const setLastOptimizationPos = useSceneStore(state => state.setLastOptimizationPos)
  const lastOptimizationConfig = useSceneStore(state => state.lastOptimizationConfig)
  const setLastOptimizationConfig = useSceneStore(state => state.setLastOptimizationConfig)
  const clearTrajectories = useSceneStore(state => state.clearTrajectories)
  
  const animationState = useAnimationStore(state => state.state)
  const totalSteps = useAnimationStore(state => state.totalSteps)
  const play = useAnimationStore(state => state.play)
  const pause = useAnimationStore(state => state.pause)
  const stop = useAnimationStore(state => state.stop)
  const setCurrentStep = useAnimationStore(state => state.setCurrentStep)
  const setTotalSteps = useAnimationStore(state => state.setTotalSteps)
  
  const enabled = useOptimizerStore(state => state.enabled)
  const getOptimizerParams = useOptimizerStore(state => state.getOptimizerParams)
  
  const pickingMode = useUIStore(state => state.pickingMode)
  const setPickingMode = useUIStore(state => state.setPickingMode)

  const deepEqual = (obj1: unknown, obj2: unknown): boolean => {
    if (obj1 === obj2) return true
    if (obj1 == null || obj2 == null) return false
    if (typeof obj1 !== 'object' || typeof obj2 !== 'object') return false
    const keys1 = Object.keys(obj1 as Record<string, unknown>)
    const keys2 = Object.keys(obj2 as Record<string, unknown>)
    if (keys1.length !== keys2.length) return false
    for (const key of keys1) {
      const val1 = (obj1 as Record<string, unknown>)[key]
      const val2 = (obj2 as Record<string, unknown>)[key]
      if (typeof val1 === 'object' && typeof val2 === 'object') {
        if (!deepEqual(val1, val2)) return false
      } else if (val1 !== val2) {
        return false
      }
    }
    return true
  }

  const hasConfigChanged = useCallback(() => {
    if (!lastOptimizationConfig) return true
    if (currentManifoldId !== lastOptimizationConfig.manifoldId) return true
    if (!deepEqual(manifoldParams, lastOptimizationConfig.manifoldParams)) return true
    if (!deepEqual(enabled, lastOptimizationConfig.optimizerEnabled)) return true
    const currentParams = getOptimizerParams()
    if (!deepEqual(currentParams, lastOptimizationConfig.optimizerParams)) return true
    return false
  }, [currentManifoldId, manifoldParams, enabled, getOptimizerParams, lastOptimizationConfig])

  const runOptimizationHandler = useCallback(async () => {
    setLoading(true, 'Computing...')
    setComputing(true)
    try {
      const optimizerParams = getOptimizerParams()
      const batchParams = optimizerParams.batch as { learningRate: number; maxIterations: number; convergenceThreshold: number; useConvergence: boolean; iterations: number }
      const momentumParams = optimizerParams.momentum as { momentum: number }
      const result = await runOptimization({
        manifold: currentManifoldId,
        initial_params: [startX, startY],
        learning_rate: batchParams.learningRate,
        momentum: momentumParams.momentum,
        n_iterations: batchParams.iterations,
        seed: 42,
        use_convergence: batchParams.useConvergence,
        max_iterations: batchParams.maxIterations,
        convergence_threshold: batchParams.convergenceThreshold,
        enabled_optimizers: enabled,
        optimizer_params: optimizerParams,
        manifold_params: manifoldParams,
      })
      const trajectories: Record<string, Array<{ x: number; y: number; z: number }>> = {}
      const convertTrajectory = (data: unknown): Array<{ x: number; y: number; z: number }> | undefined => {
        if (!data || !Array.isArray(data)) return undefined
        return data.map((point: number[] | { x: number; y: number; z: number }) => {
          if (Array.isArray(point)) return { x: point[0], y: point[1], z: point[2] }
          return point
        })
      }
      const batchData = convertTrajectory(result.batch)
      const momentumData = convertTrajectory(result.momentum)
      const adamData = convertTrajectory(result.adam)
      const sgdData = convertTrajectory(result.sgd)
      const wheelData = convertTrajectory(result.wheel)
      if (batchData) trajectories.batch = batchData
      if (momentumData) trajectories.momentum = momentumData
      if (adamData) trajectories.adam = adamData
      if (sgdData) trajectories.sgd = sgdData
      if (wheelData) trajectories.wheel = wheelData
      setTrajectories(trajectories)
      let maxLength = 0
      Object.values(trajectories).forEach(trajectory => {
        if (trajectory && trajectory.length > maxLength) maxLength = trajectory.length
      })
      setTotalSteps(maxLength)
      setCurrentStep(0)
      setLastOptimizationPos(startX, startY)
      setLastOptimizationConfig({
        manifoldId: currentManifoldId,
        manifoldParams: { ...manifoldParams },
        optimizerEnabled: { ...enabled },
        optimizerParams: getOptimizerParams()
      })
    } catch {
      // Error handling
    } finally {
      setLoading(false)
      setComputing(false)
    }
  }, [currentManifoldId, startX, startY, enabled, manifoldParams, getOptimizerParams, setTrajectories, setTotalSteps, setCurrentStep, setLoading, setComputing, setLastOptimizationPos, setLastOptimizationConfig])

  const handlePlayPause = useCallback(async () => {
    if (isComputing) return
    if (animationState === 'stopped' || animationState === 'paused') {
      const threshold = 0.01
      const positionChanged = !lastOptimizationStartPos || 
        Math.abs(startX - lastOptimizationStartPos.x) > threshold ||
        Math.abs(startY - lastOptimizationStartPos.y) > threshold
      if (totalSteps === 0 || positionChanged || hasConfigChanged()) {
        stop()
        clearTrajectories()
        await runOptimizationHandler()
      } else {
        setCurrentStep(0)
      }
      play()
    } else {
      pause()
    }
  }, [isComputing, animationState, totalSteps, startX, startY, lastOptimizationStartPos, play, pause, stop, clearTrajectories, runOptimizationHandler, hasConfigChanged, setCurrentStep])

  const handlePickClick = () => {
    if (animationState !== 'playing' && !isComputing) {
      setPickingMode(!pickingMode)
    }
  }

  return (
    <div className={styles.floatingControls}>
      <button 
        className={`${styles.playBtn} ${animationState === 'playing' ? styles.playing : ''}`}
        onClick={handlePlayPause}
        disabled={isComputing}
        aria-label={isComputing ? 'Computing' : animationState === 'playing' ? 'Pause' : 'Play'}
      >
        {isComputing ? (
          <svg className={styles.spinner} viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="63" strokeDashoffset="0">
              <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
            </circle>
          </svg>
        ) : animationState === 'playing' ? (
          <svg viewBox="0 0 24 24">
            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" fill="currentColor"/>
          </svg>
        ) : (
          <svg viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" fill="currentColor"/>
          </svg>
        )}
      </button>
      
      <button 
        className={`${styles.pickBtn} ${pickingMode ? styles.active : ''}`}
        onClick={handlePickClick}
        disabled={animationState === 'playing' || isComputing}
        aria-label="Pick starting point"
      >
        <svg viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="2" fill="currentColor"/>
          <circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" strokeWidth="2"/>
          <line x1="12" y1="2" x2="12" y2="7" stroke="currentColor" strokeWidth="2"/>
          <line x1="12" y1="17" x2="12" y2="22" stroke="currentColor" strokeWidth="2"/>
          <line x1="2" y1="12" x2="7" y2="12" stroke="currentColor" strokeWidth="2"/>
          <line x1="17" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </button>
    </div>
  )
}
