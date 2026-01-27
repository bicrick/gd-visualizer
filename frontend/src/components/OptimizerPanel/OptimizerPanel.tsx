import { useEffect, useCallback } from 'react'
import { useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import styles from './OptimizerPanel.module.css'

interface SliderControlProps {
  label: string
  tooltip?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  format?: (value: number) => string
}

function SliderControl({ label, value, min, max, step, onChange, format }: SliderControlProps) {
  const displayValue = format ? format(value) : value.toFixed(3)
  
  return (
    <div className={styles.control}>
      <div className={styles.controlLabel}>
        <span>{label}</span>
        <span className={styles.controlValue}>{displayValue}</span>
      </div>
      <input
        type="range"
        className={styles.slider}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  )
}

const OPTIMIZER_DISPLAY_NAMES: Record<string, string> = {
  batch: 'Batch GD',
  momentum: 'Momentum GD',
  adam: 'ADAM',
  sgd: 'SGD (Stochastic)',
}

export function OptimizerPanel() {
  const activePanelOptimizer = useOptimizerStore(state => state.activePanelOptimizer)
  const closeOptimizerPanel = useOptimizerStore(state => state.closeOptimizerPanel)
  const batch = useOptimizerStore(state => state.batch)
  const momentum = useOptimizerStore(state => state.momentum)
  const adam = useOptimizerStore(state => state.adam)
  const sgd = useOptimizerStore(state => state.sgd)
  const setOptimizerParam = useOptimizerStore(state => state.setOptimizerParam)
  
  // Close on escape key
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      closeOptimizerPanel()
    }
  }, [closeOptimizerPanel])
  
  useEffect(() => {
    if (activePanelOptimizer) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [activePanelOptimizer, handleKeyDown])
  
  if (!activePanelOptimizer) return null
  
  const displayName = OPTIMIZER_DISPLAY_NAMES[activePanelOptimizer]
  const color = OPTIMIZER_COLORS[activePanelOptimizer]
  
  const renderBatchControls = () => (
    <>
      <SliderControl
        label="Learning Rate"
        value={batch.learningRate}
        min={0.001}
        max={0.1}
        step={0.001}
        onChange={(v) => setOptimizerParam('batch', 'learningRate', v)}
      />
      <div className={styles.checkboxControl}>
        <label>
          <input
            type="checkbox"
            checked={batch.useConvergence}
            onChange={(e) => setOptimizerParam('batch', 'useConvergence', e.target.checked)}
          />
          Run Until Convergence
        </label>
      </div>
      {batch.useConvergence ? (
        <>
          <SliderControl
            label="Max Iterations"
            value={batch.maxIterations}
            min={500}
            max={20000}
            step={500}
            onChange={(v) => setOptimizerParam('batch', 'maxIterations', v)}
            format={(v) => v.toString()}
          />
          <SliderControl
            label="Convergence Threshold"
            value={batch.convergenceThreshold}
            min={0.000001}
            max={0.01}
            step={0.000001}
            onChange={(v) => setOptimizerParam('batch', 'convergenceThreshold', v)}
            format={(v) => v.toExponential(0)}
          />
        </>
      ) : (
        <SliderControl
          label="Iterations"
          value={batch.iterations}
          min={50}
          max={500}
          step={10}
          onChange={(v) => setOptimizerParam('batch', 'iterations', v)}
          format={(v) => v.toString()}
        />
      )}
    </>
  )
  
  const renderMomentumControls = () => (
    <>
      <SliderControl
        label="Learning Rate"
        value={momentum.learningRate}
        min={0.001}
        max={0.1}
        step={0.001}
        onChange={(v) => setOptimizerParam('momentum', 'learningRate', v)}
      />
      <SliderControl
        label="Momentum"
        value={momentum.momentum}
        min={0}
        max={0.99}
        step={0.01}
        onChange={(v) => setOptimizerParam('momentum', 'momentum', v)}
        format={(v) => v.toFixed(2)}
      />
      <SliderControl
        label="LR Decay"
        value={momentum.lrDecay}
        min={0.95}
        max={1.0}
        step={0.001}
        onChange={(v) => setOptimizerParam('momentum', 'lrDecay', v)}
      />
      <div className={styles.checkboxControl}>
        <label>
          <input
            type="checkbox"
            checked={momentum.useConvergence}
            onChange={(e) => setOptimizerParam('momentum', 'useConvergence', e.target.checked)}
          />
          Run Until Convergence
        </label>
      </div>
    </>
  )
  
  const renderAdamControls = () => (
    <>
      <SliderControl
        label="Learning Rate"
        value={adam.learningRate}
        min={0.001}
        max={0.1}
        step={0.001}
        onChange={(v) => setOptimizerParam('adam', 'learningRate', v)}
      />
      <SliderControl
        label="Beta1"
        value={adam.beta1}
        min={0.8}
        max={0.999}
        step={0.001}
        onChange={(v) => setOptimizerParam('adam', 'beta1', v)}
      />
      <SliderControl
        label="Beta2"
        value={adam.beta2}
        min={0.9}
        max={0.9999}
        step={0.0001}
        onChange={(v) => setOptimizerParam('adam', 'beta2', v)}
        format={(v) => v.toFixed(4)}
      />
      <SliderControl
        label="Epsilon"
        value={adam.epsilon}
        min={0.00000001}
        max={0.0001}
        step={0.00000001}
        onChange={(v) => setOptimizerParam('adam', 'epsilon', v)}
        format={(v) => v.toExponential(0)}
      />
      <div className={styles.checkboxControl}>
        <label>
          <input
            type="checkbox"
            checked={adam.useConvergence}
            onChange={(e) => setOptimizerParam('adam', 'useConvergence', e.target.checked)}
          />
          Run Until Convergence
        </label>
      </div>
    </>
  )
  
  const renderSgdControls = () => (
    <>
      <SliderControl
        label="Learning Rate"
        value={sgd.learningRate}
        min={0.001}
        max={0.1}
        step={0.001}
        onChange={(v) => setOptimizerParam('sgd', 'learningRate', v)}
      />
      <SliderControl
        label="Step Multiplier"
        value={sgd.stepMultiplier}
        min={1.0}
        max={6.0}
        step={0.5}
        onChange={(v) => setOptimizerParam('sgd', 'stepMultiplier', v)}
        format={(v) => v.toFixed(1) + 'x'}
      />
      <div className={styles.paramHint}>
        Effective step size multiplier (higher = faster convergence)
      </div>
      <SliderControl
        label="Noise Scale"
        value={sgd.noiseScale}
        min={0.1}
        max={2.0}
        step={0.1}
        onChange={(v) => setOptimizerParam('sgd', 'noiseScale', v)}
        format={(v) => v.toFixed(1)}
      />
      <div className={styles.paramHint}>
        Initial gradient noise magnitude (higher = more radical bouncing)
      </div>
      <SliderControl
        label="Noise Decay"
        value={sgd.noiseDecay}
        min={0.95}
        max={1.0}
        step={0.005}
        onChange={(v) => setOptimizerParam('sgd', 'noiseDecay', v)}
        format={(v) => v.toFixed(3)}
      />
      <div className={styles.paramHint}>
        Noise reduction per step (lower = faster settling, 1.0 = no decay)
      </div>
      <div className={styles.checkboxControl}>
        <label>
          <input
            type="checkbox"
            checked={sgd.useConvergence}
            onChange={(e) => setOptimizerParam('sgd', 'useConvergence', e.target.checked)}
          />
          Run Until Convergence
        </label>
      </div>
    </>
  )
  
  const renderControls = () => {
    switch (activePanelOptimizer) {
      case 'batch':
        return renderBatchControls()
      case 'momentum':
        return renderMomentumControls()
      case 'adam':
        return renderAdamControls()
      case 'sgd':
        return renderSgdControls()
      default:
        return null
    }
  }
  
  return (
    <div className={styles.overlay} onClick={closeOptimizerPanel}>
      <div className={styles.panel} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.titleContainer}>
            <span className={styles.colorIndicator} style={{ backgroundColor: color }} />
            <h2 className={styles.title}>{displayName}</h2>
          </div>
          <button className={styles.closeBtn} onClick={closeOptimizerPanel}>
            <svg viewBox="0 0 24 24" className={styles.closeIcon}>
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className={styles.content}>
          {renderControls()}
        </div>
      </div>
    </div>
  )
}
