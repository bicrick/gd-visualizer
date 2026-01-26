import { useCallback, useEffect } from 'react'
import { Card } from '../Card'
import { useSceneStore, useAnimationStore, useOptimizerStore, useUIStore } from '../../stores'
import { runOptimization } from '../../utils/api'
import styles from './AnimationCard.module.css'

export function AnimationCard() {
  const startX = useSceneStore(state => state.startX)
  const startY = useSceneStore(state => state.startY)
  const setStartPosition = useSceneStore(state => state.setStartPosition)
  const randomizeStartPosition = useSceneStore(state => state.randomizeStartPosition)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const manifoldParams = useSceneStore(state => state.manifoldParams)
  const setTrajectories = useSceneStore(state => state.setTrajectories)
  const setLoading = useSceneStore(state => state.setLoading)
  const isComputing = useSceneStore(state => state.isComputing)
  const setComputing = useSceneStore(state => state.setComputing)
  const lastOptimizationStartPos = useSceneStore(state => state.lastOptimizationStartPos)
  const setLastOptimizationPos = useSceneStore(state => state.setLastOptimizationPos)
  const clearTrajectories = useSceneStore(state => state.clearTrajectories)
  
  const animationState = useAnimationStore(state => state.state)
  const currentStep = useAnimationStore(state => state.currentStep)
  const totalSteps = useAnimationStore(state => state.totalSteps)
  const speed = useAnimationStore(state => state.speed)
  const play = useAnimationStore(state => state.play)
  const pause = useAnimationStore(state => state.pause)
  const stop = useAnimationStore(state => state.stop)
  const setCurrentStep = useAnimationStore(state => state.setCurrentStep)
  const setTotalSteps = useAnimationStore(state => state.setTotalSteps)
  const setSpeed = useAnimationStore(state => state.setSpeed)
  
  const enabled = useOptimizerStore(state => state.enabled)
  const getOptimizerParams = useOptimizerStore(state => state.getOptimizerParams)
  
  const pickingMode = useUIStore(state => state.pickingMode)
  const setPickingMode = useUIStore(state => state.setPickingMode)
  
  // Exit picking mode if animation is playing
  useEffect(() => {
    if (animationState === 'playing' && pickingMode) {
      setPickingMode(false)
    }
  }, [animationState, pickingMode, setPickingMode])
  
  const runOptimizationHandler = useCallback(async () => {
    setLoading(true, 'Computing trajectories...')
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
      
      // Extract trajectories from response (they're at root level, not nested)
      // Backend returns arrays of [x, y, z] tuples, convert to {x, y, z} objects
      const trajectories: Record<string, Array<{ x: number; y: number; z: number }>> = {}
      
      const convertTrajectory = (data: unknown): Array<{ x: number; y: number; z: number }> | undefined => {
        if (!data || !Array.isArray(data)) return undefined
        return data.map((point: number[] | { x: number; y: number; z: number }) => {
          // Handle both array format [x, y, z] and object format {x, y, z}
          if (Array.isArray(point)) {
            return { x: point[0], y: point[1], z: point[2] }
          }
          return point
        })
      }
      
      const batchData = convertTrajectory(result.batch)
      const momentumData = convertTrajectory(result.momentum)
      const adamData = convertTrajectory(result.adam)
      const sgdData = convertTrajectory(result.sgd)
      
      if (batchData) trajectories.batch = batchData
      if (momentumData) trajectories.momentum = momentumData
      if (adamData) trajectories.adam = adamData
      if (sgdData) trajectories.sgd = sgdData
      
      setTrajectories(trajectories)
      
      // Calculate max trajectory length
      let maxLength = 0
      Object.values(trajectories).forEach(trajectory => {
        if (trajectory && trajectory.length > maxLength) {
          maxLength = trajectory.length
        }
      })
      setTotalSteps(maxLength)
      setCurrentStep(0)
      
      // Save the position used for this optimization
      setLastOptimizationPos(startX, startY)
      
    } catch (error) {
      console.error('Optimization failed:', error)
    } finally {
      setLoading(false)
      setComputing(false)
    }
  }, [currentManifoldId, startX, startY, enabled, manifoldParams, getOptimizerParams, setTrajectories, setTotalSteps, setCurrentStep, setLoading, setComputing, setLastOptimizationPos])
  
  const handlePlayPause = useCallback(async () => {
    // Prevent spam clicks - if already computing, do nothing
    if (isComputing) {
      return
    }
    
    if (animationState === 'stopped' || animationState === 'paused') {
      // Check if we need to recompute trajectories
      const threshold = 0.01
      const positionChanged = !lastOptimizationStartPos || 
        Math.abs(startX - lastOptimizationStartPos.x) > threshold ||
        Math.abs(startY - lastOptimizationStartPos.y) > threshold
      
      // Run optimization if no trajectories exist OR position has changed
      if (totalSteps === 0 || positionChanged) {
        // Clear old trajectories and reset animation state before computing new ones
        stop()
        clearTrajectories()
        await runOptimizationHandler()
      }
      play()
    } else {
      pause()
    }
  }, [isComputing, animationState, totalSteps, startX, startY, lastOptimizationStartPos, play, pause, stop, clearTrajectories, runOptimizationHandler])
  
  
  const summaryText = isComputing
    ? 'Computing...'
    : animationState === 'playing' 
    ? `Playing ${currentStep}/${totalSteps}`
    : animationState === 'paused'
    ? `Paused ${currentStep}/${totalSteps}`
    : totalSteps > 0 
    ? `Ready ${totalSteps} steps`
    : 'Stopped'
  
  const collapsedControls = (
    <>
      <button 
        className={`${styles.iconBtn} ${styles.playBtn}`}
        onClick={handlePlayPause}
        disabled={isComputing}
        title={isComputing ? 'Computing...' : animationState === 'playing' ? 'Pause' : 'Play'}
      >
        {isComputing ? (
          <svg className={styles.iconSmall} viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="9" fill="none" stroke="white" strokeWidth="2" strokeDasharray="57" strokeDashoffset="0">
              <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
            </circle>
          </svg>
        ) : animationState === 'playing' ? (
          <svg className={styles.iconSmall} viewBox="0 0 24 24">
            <path d="M6 5h4v14H6V5zm8 0h4v14h-4V5z"/>
          </svg>
        ) : (
          <svg className={styles.iconSmall} viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z"/>
          </svg>
        )}
      </button>
      <button 
        className={`${styles.iconBtn} ${styles.pickBtn}`}
        onClick={() => setPickingMode(true)}
        disabled={animationState === 'playing' || isComputing}
        title="Pick starting point"
      >
        <svg className={styles.iconSmall} viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="2" fill="white"/>
          <circle cx="12" cy="12" r="7" fill="none" stroke="white" strokeWidth="2"/>
          <line x1="12" y1="2" x2="12" y2="7" stroke="white" strokeWidth="2"/>
          <line x1="12" y1="17" x2="12" y2="22" stroke="white" strokeWidth="2"/>
          <line x1="2" y1="12" x2="7" y2="12" stroke="white" strokeWidth="2"/>
          <line x1="17" y1="12" x2="22" y2="12" stroke="white" strokeWidth="2"/>
        </svg>
      </button>
    </>
  )
  
  return (
    <Card 
      title="Animation" 
      summary={summaryText}
      defaultCollapsed={false}
      collapsedControls={collapsedControls}
    >
      <div className={styles.content}>
        {/* Starting Position */}
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Starting Position</h3>
          <div className={styles.positionInputs}>
            <div className={styles.inputGroup}>
              <label className={styles.label}>X:</label>
              <input
                type="number"
                className={styles.input}
                value={startX.toFixed(2)}
                onChange={(e) => setStartPosition(parseFloat(e.target.value), startY)}
                step={0.1}
                disabled={animationState === 'playing' || isComputing}
              />
            </div>
            <div className={styles.inputGroup}>
              <label className={styles.label}>Y:</label>
              <input
                type="number"
                className={styles.input}
                value={startY.toFixed(2)}
                onChange={(e) => setStartPosition(startX, parseFloat(e.target.value))}
                step={0.1}
                disabled={animationState === 'playing' || isComputing}
              />
            </div>
          </div>
          <div className={styles.buttonRow}>
            <button 
              className={styles.smallButton}
              onClick={randomizeStartPosition}
              disabled={animationState === 'playing' || isComputing}
            >
              Random
            </button>
            <button 
              className={styles.smallButton}
              onClick={() => setPickingMode(true)}
              disabled={animationState === 'playing' || isComputing}
            >
              Pick Point
            </button>
          </div>
        </div>
        
        {/* Media Controls */}
        <div className={styles.section}>
          <div className={styles.mediaControls}>
            <button 
              className={`${styles.mediaBtn} ${styles.primary}`}
              onClick={handlePlayPause}
              disabled={isComputing}
              title={isComputing ? 'Computing...' : animationState === 'playing' ? 'Pause' : 'Play'}
            >
              {isComputing ? (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" fill="none" stroke="white" strokeWidth="2" strokeDasharray="63" strokeDashoffset="0">
                    <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
                  </circle>
                </svg>
              ) : animationState === 'playing' ? (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                </svg>
              ) : (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z"/>
                </svg>
              )}
            </button>
          </div>
        </div>
        
        {/* Timeline */}
        <div className={styles.section}>
          <div className={styles.sliderGroup}>
            <div className={styles.sliderLabel}>
              <span>Iteration</span>
              <span className={styles.sliderValue}>
                <span className={styles.currentStep}>{currentStep}</span>
                <span className={styles.separator}>/</span>
                <span className={styles.totalSteps}>{totalSteps}</span>
              </span>
            </div>
            <input
              type="range"
              className={styles.slider}
              min={0}
              max={totalSteps || 1}
              value={currentStep}
              onChange={(e) => setCurrentStep(parseInt(e.target.value))}
              disabled={totalSteps === 0}
            />
          </div>
          
          <div className={styles.sliderGroup}>
            <div className={styles.sliderLabel}>
              <span>Speed</span>
              <span className={styles.sliderValue}>{speed.toFixed(1)}</span>
            </div>
            <input
              type="range"
              className={styles.slider}
              min={0.1}
              max={5}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>
    </Card>
  )
}
