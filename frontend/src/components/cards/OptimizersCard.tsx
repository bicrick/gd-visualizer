import { useState } from 'react'
import { Card } from '../Card'
import { useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import styles from './OptimizersCard.module.css'

interface OptimizerSectionProps {
  displayName: string
  color: string
  enabled: boolean
  onToggle: (enabled: boolean) => void
  children: React.ReactNode
}

function OptimizerSection({ displayName, color, enabled, onToggle, children }: OptimizerSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  
  return (
    <div className={styles.section}>
      <div 
        className={styles.header}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <label className={styles.toggleLabel} onClick={(e) => e.stopPropagation()}>
          <input
            type="checkbox"
            className={styles.checkbox}
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span className={styles.colorIndicator} style={{ backgroundColor: color }} />
          <span className={styles.optimizerName}>{displayName}</span>
        </label>
        <span className={`${styles.expandBtn} ${isExpanded ? '' : styles.collapsed}`}>
          &#9660;
        </span>
      </div>
      
      <div className={`${styles.params} ${isExpanded ? '' : styles.paramsCollapsed}`}>
        {children}
      </div>
    </div>
  )
}

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

export function OptimizersCard() {
  const enabled = useOptimizerStore(state => state.enabled)
  const batch = useOptimizerStore(state => state.batch)
  const momentum = useOptimizerStore(state => state.momentum)
  const adam = useOptimizerStore(state => state.adam)
  const sgd = useOptimizerStore(state => state.sgd)
  const toggleOptimizer = useOptimizerStore(state => state.toggleOptimizer)
  const setOptimizerParam = useOptimizerStore(state => state.setOptimizerParam)
  
  const enabledCount = Object.values(enabled).filter(Boolean).length
  const totalCount = Object.keys(enabled).length
  
  return (
    <Card 
      title="Optimizers" 
      summary={`${enabledCount}/${totalCount} enabled`}
      defaultCollapsed={true}
    >
      <div className={styles.content} data-scrollable="true">
        {/* Batch GD */}
        <OptimizerSection
          displayName="Batch GD"
          color={OPTIMIZER_COLORS.batch}
          enabled={enabled.batch}
          onToggle={(v) => toggleOptimizer('batch', v)}
        >
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
        </OptimizerSection>
        
        {/* Momentum GD */}
        <OptimizerSection
          displayName="Momentum GD"
          color={OPTIMIZER_COLORS.momentum}
          enabled={enabled.momentum}
          onToggle={(v) => toggleOptimizer('momentum', v)}
        >
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
        </OptimizerSection>
        
        {/* ADAM */}
        <OptimizerSection
          displayName="ADAM"
          color={OPTIMIZER_COLORS.adam}
          enabled={enabled.adam}
          onToggle={(v) => toggleOptimizer('adam', v)}
        >
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
        </OptimizerSection>
        
        {/* SGD */}
        <OptimizerSection
          displayName="SGD (Stochastic)"
          color={OPTIMIZER_COLORS.sgd}
          enabled={enabled.sgd}
          onToggle={(v) => toggleOptimizer('sgd', v)}
        >
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
          <SliderControl
            label="Noise Scale"
            value={sgd.noiseScale}
            min={0.1}
            max={2.0}
            step={0.1}
            onChange={(v) => setOptimizerParam('sgd', 'noiseScale', v)}
            format={(v) => v.toFixed(1)}
          />
          <SliderControl
            label="Noise Decay"
            value={sgd.noiseDecay}
            min={0.95}
            max={1.0}
            step={0.005}
            onChange={(v) => setOptimizerParam('sgd', 'noiseDecay', v)}
            format={(v) => v.toFixed(3)}
          />
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
        </OptimizerSection>
      </div>
    </Card>
  )
}
