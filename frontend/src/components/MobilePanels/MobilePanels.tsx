import { useEffect, useState } from 'react'
import { MobileBottomSheet } from '../MobileBottomSheet'
import { useUIStore } from '../../stores/uiStore'
import { useSceneStore, useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import { fetchManifolds } from '../../utils/api'
import styles from './MobilePanels.module.css'

// Manifold Panel Content
function ManifoldPanelContent() {
  const manifolds = useSceneStore(state => state.manifolds)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const setManifolds = useSceneStore(state => state.setManifolds)
  const setCurrentManifold = useSceneStore(state => state.setCurrentManifold)
  const setActiveMobilePanel = useUIStore(state => state.setActiveMobilePanel)

  useEffect(() => {
    if (manifolds.length === 0) {
      fetchManifolds()
        .then(data => {
          setManifolds(data.manifolds)
        })
        .catch(_err => {
          // Error handling
        })
    }
  }, [manifolds.length, setManifolds])

  const handleSelect = (id: string) => {
    setCurrentManifold(id)
    setActiveMobilePanel(null)
  }

  return (
    <div className={styles.options}>
      {manifolds.map(manifold => (
        <button
          key={manifold.id}
          className={`${styles.option} ${manifold.id === currentManifoldId ? styles.selected : ''}`}
          onClick={() => handleSelect(manifold.id)}
        >
          <div className={styles.optionName}>{manifold.name}</div>
          <div className={styles.optionDescription}>{manifold.description}</div>
        </button>
      ))}
    </div>
  )
}

// Parameters Panel Content
function ParamsPanelContent() {
  const manifolds = useSceneStore(state => state.manifolds)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const manifoldParams = useSceneStore(state => state.manifoldParams)
  const setManifoldParam = useSceneStore(state => state.setManifoldParam)
  const setManifolds = useSceneStore(state => state.setManifolds)

  // Fetch manifolds if not loaded yet
  useEffect(() => {
    if (manifolds.length === 0) {
      fetchManifolds()
        .then(data => {
          setManifolds(data.manifolds)
        })
        .catch(_err => {
          // Error handling
        })
    }
  }, [manifolds.length, setManifolds])

  const currentManifold = manifolds.find(m => m.id === currentManifoldId)
  const parameters = currentManifold?.parameters || []

  if (parameters.length === 0) {
    return (
      <div className={styles.emptyState}>
        <p>No parameters available for this manifold.</p>
      </div>
    )
  }

  return (
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
              className={styles.slider}
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
  )
}

// Optimizers Panel Content
function OptimizersPanelContent() {
  const [expandedOptimizer, setExpandedOptimizer] = useState<string | null>(null)
  const enabled = useOptimizerStore(state => state.enabled)
  const batch = useOptimizerStore(state => state.batch)
  const momentum = useOptimizerStore(state => state.momentum)
  const adam = useOptimizerStore(state => state.adam)
  const sgd = useOptimizerStore(state => state.sgd)
  const wheel = useOptimizerStore(state => state.wheel)
  const toggleOptimizer = useOptimizerStore(state => state.toggleOptimizer)
  const setOptimizerParam = useOptimizerStore(state => state.setOptimizerParam)

  const optimizers = [
    { id: 'batch', name: 'Batch GD', params: batch },
    { id: 'momentum', name: 'Momentum GD', params: momentum },
    { id: 'adam', name: 'ADAM', params: adam },
    { id: 'sgd', name: 'SGD (Stochastic)', params: sgd },
    { id: 'wheel', name: 'Wheel', params: wheel },
  ] as const

  const toggleExpand = (id: string) => {
    setExpandedOptimizer(expandedOptimizer === id ? null : id)
  }

  return (
    <div className={styles.optimizerList}>
      {optimizers.map(optimizer => (
        <div key={optimizer.id} className={styles.optimizerSection}>
          <div className={styles.optimizerHeader}>
            <label 
              className={styles.optimizerItem}
              onClick={(e) => {
                e.stopPropagation()
              }}
            >
              <input
                type="checkbox"
                className={styles.checkbox}
                checked={enabled[optimizer.id]}
                onChange={(e) => toggleOptimizer(optimizer.id, e.target.checked)}
              />
              <span 
                className={styles.colorDot} 
                style={{ backgroundColor: OPTIMIZER_COLORS[optimizer.id] }}
              />
              <span className={styles.optimizerName}>{optimizer.name}</span>
            </label>
            <button
              className={styles.expandBtn}
              onClick={() => toggleExpand(optimizer.id)}
              aria-label={expandedOptimizer === optimizer.id ? 'Collapse' : 'Expand'}
            >
              {expandedOptimizer === optimizer.id ? '▼' : '▶'}
            </button>
          </div>
          
          {expandedOptimizer === optimizer.id && (
            <div className={styles.optimizerParams}>
              {optimizer.id === 'batch' && (
                <>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Learning Rate</span>
                      <span className={styles.paramValue}>{batch.learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      value={batch.learningRate}
                      onChange={(e) => setOptimizerParam('batch', 'learningRate', parseFloat(e.target.value))}
                    />
                  </div>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={batch.useConvergence}
                      onChange={(e) => setOptimizerParam('batch', 'useConvergence', e.target.checked)}
                    />
                    Run Until Convergence
                  </label>
                  {batch.useConvergence ? (
                    <>
                      <div className={styles.paramControl}>
                        <div className={styles.paramLabel}>
                          <span className={styles.paramName}>Max Iterations</span>
                          <span className={styles.paramValue}>{batch.maxIterations}</span>
                        </div>
                        <input
                          type="range"
                          className={styles.slider}
                          min={500}
                          max={20000}
                          step={500}
                          value={batch.maxIterations}
                          onChange={(e) => setOptimizerParam('batch', 'maxIterations', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className={styles.paramControl}>
                        <div className={styles.paramLabel}>
                          <span className={styles.paramName}>Convergence Threshold</span>
                          <span className={styles.paramValue}>{batch.convergenceThreshold.toExponential(0)}</span>
                        </div>
                        <input
                          type="range"
                          className={styles.slider}
                          min={0.000001}
                          max={0.01}
                          step={0.000001}
                          value={batch.convergenceThreshold}
                          onChange={(e) => setOptimizerParam('batch', 'convergenceThreshold', parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  ) : (
                    <div className={styles.paramControl}>
                      <div className={styles.paramLabel}>
                        <span className={styles.paramName}>Iterations</span>
                        <span className={styles.paramValue}>{batch.iterations}</span>
                      </div>
                      <input
                        type="range"
                        className={styles.slider}
                        min={50}
                        max={500}
                        step={10}
                        value={batch.iterations}
                        onChange={(e) => setOptimizerParam('batch', 'iterations', parseFloat(e.target.value))}
                      />
                    </div>
                  )}
                </>
              )}
              
              {optimizer.id === 'momentum' && (
                <>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Learning Rate</span>
                      <span className={styles.paramValue}>{momentum.learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      value={momentum.learningRate}
                      onChange={(e) => setOptimizerParam('momentum', 'learningRate', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Momentum</span>
                      <span className={styles.paramValue}>{momentum.momentum.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0}
                      max={0.99}
                      step={0.01}
                      value={momentum.momentum}
                      onChange={(e) => setOptimizerParam('momentum', 'momentum', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>LR Decay</span>
                      <span className={styles.paramValue}>{momentum.lrDecay.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.95}
                      max={1.0}
                      step={0.001}
                      value={momentum.lrDecay}
                      onChange={(e) => setOptimizerParam('momentum', 'lrDecay', parseFloat(e.target.value))}
                    />
                  </div>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={momentum.useConvergence}
                      onChange={(e) => setOptimizerParam('momentum', 'useConvergence', e.target.checked)}
                    />
                    Run Until Convergence
                  </label>
                </>
              )}
              
              {optimizer.id === 'adam' && (
                <>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Learning Rate</span>
                      <span className={styles.paramValue}>{adam.learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      value={adam.learningRate}
                      onChange={(e) => setOptimizerParam('adam', 'learningRate', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Beta1</span>
                      <span className={styles.paramValue}>{adam.beta1.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.8}
                      max={0.999}
                      step={0.001}
                      value={adam.beta1}
                      onChange={(e) => setOptimizerParam('adam', 'beta1', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Beta2</span>
                      <span className={styles.paramValue}>{adam.beta2.toFixed(4)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.9}
                      max={0.9999}
                      step={0.0001}
                      value={adam.beta2}
                      onChange={(e) => setOptimizerParam('adam', 'beta2', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Epsilon</span>
                      <span className={styles.paramValue}>{adam.epsilon.toExponential(0)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.00000001}
                      max={0.0001}
                      step={0.00000001}
                      value={adam.epsilon}
                      onChange={(e) => setOptimizerParam('adam', 'epsilon', parseFloat(e.target.value))}
                    />
                  </div>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={adam.useConvergence}
                      onChange={(e) => setOptimizerParam('adam', 'useConvergence', e.target.checked)}
                    />
                    Run Until Convergence
                  </label>
                </>
              )}
              
              {optimizer.id === 'sgd' && (
                <>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Learning Rate</span>
                      <span className={styles.paramValue}>{sgd.learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      value={sgd.learningRate}
                      onChange={(e) => setOptimizerParam('sgd', 'learningRate', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Step Multiplier</span>
                      <span className={styles.paramValue}>{sgd.stepMultiplier.toFixed(1)}x</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={1.0}
                      max={6.0}
                      step={0.5}
                      value={sgd.stepMultiplier}
                      onChange={(e) => setOptimizerParam('sgd', 'stepMultiplier', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Noise Scale</span>
                      <span className={styles.paramValue}>{sgd.noiseScale.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.1}
                      max={2.0}
                      step={0.1}
                      value={sgd.noiseScale}
                      onChange={(e) => setOptimizerParam('sgd', 'noiseScale', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Noise Decay</span>
                      <span className={styles.paramValue}>{sgd.noiseDecay.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.95}
                      max={1.0}
                      step={0.005}
                      value={sgd.noiseDecay}
                      onChange={(e) => setOptimizerParam('sgd', 'noiseDecay', parseFloat(e.target.value))}
                    />
                  </div>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={sgd.useConvergence}
                      onChange={(e) => setOptimizerParam('sgd', 'useConvergence', e.target.checked)}
                    />
                    Run Until Convergence
                  </label>
                </>
              )}
              
              {optimizer.id === 'wheel' && (
                <>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Learning Rate</span>
                      <span className={styles.paramValue}>{wheel.learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      value={wheel.learningRate}
                      onChange={(e) => setOptimizerParam('wheel', 'learningRate', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Beta (Momentum Decay)</span>
                      <span className={styles.paramValue}>{wheel.beta.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.8}
                      max={0.99}
                      step={0.01}
                      value={wheel.beta}
                      onChange={(e) => setOptimizerParam('wheel', 'beta', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className={styles.paramControl}>
                    <div className={styles.paramLabel}>
                      <span className={styles.paramName}>Moment of Inertia</span>
                      <span className={styles.paramValue}>{wheel.momentOfInertia.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      className={styles.slider}
                      min={0.1}
                      max={15}
                      step={0.1}
                      value={wheel.momentOfInertia}
                      onChange={(e) => setOptimizerParam('wheel', 'momentOfInertia', parseFloat(e.target.value))}
                    />
                  </div>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={wheel.useConvergence}
                      onChange={(e) => setOptimizerParam('wheel', 'useConvergence', e.target.checked)}
                    />
                    Run Until Convergence
                  </label>
                  {wheel.useConvergence ? (
                    <>
                      <div className={styles.paramControl}>
                        <div className={styles.paramLabel}>
                          <span className={styles.paramName}>Max Iterations</span>
                          <span className={styles.paramValue}>{wheel.maxIterations}</span>
                        </div>
                        <input
                          type="range"
                          className={styles.slider}
                          min={500}
                          max={20000}
                          step={500}
                          value={wheel.maxIterations}
                          onChange={(e) => setOptimizerParam('wheel', 'maxIterations', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className={styles.paramControl}>
                        <div className={styles.paramLabel}>
                          <span className={styles.paramName}>Convergence Threshold</span>
                          <span className={styles.paramValue}>{wheel.convergenceThreshold.toExponential(0)}</span>
                        </div>
                        <input
                          type="range"
                          className={styles.slider}
                          min={0.000001}
                          max={0.01}
                          step={0.000001}
                          value={wheel.convergenceThreshold}
                          onChange={(e) => setOptimizerParam('wheel', 'convergenceThreshold', parseFloat(e.target.value))}
                        />
                      </div>
                    </>
                  ) : (
                    <div className={styles.paramControl}>
                      <div className={styles.paramLabel}>
                        <span className={styles.paramName}>Iterations</span>
                        <span className={styles.paramValue}>{wheel.iterations}</span>
                      </div>
                      <input
                        type="range"
                        className={styles.slider}
                        min={50}
                        max={500}
                        step={10}
                        value={wheel.iterations}
                        onChange={(e) => setOptimizerParam('wheel', 'iterations', parseFloat(e.target.value))}
                      />
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// Main Mobile Panels Component
export function MobilePanels() {
  const activeMobilePanel = useUIStore(state => state.activeMobilePanel)

  if (!activeMobilePanel) return null

  const panelConfig = {
    manifold: { title: 'Select Manifold', content: <ManifoldPanelContent /> },
    params: { title: 'Manifold Parameters', content: <ParamsPanelContent /> },
    optimizers: { title: 'Optimizers', content: <OptimizersPanelContent /> },
  }

  const config = panelConfig[activeMobilePanel]

  return (
    <MobileBottomSheet title={config.title}>
      {config.content}
    </MobileBottomSheet>
  )
}
