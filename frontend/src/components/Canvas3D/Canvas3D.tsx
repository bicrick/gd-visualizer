import { useMemo, useCallback, useRef } from 'react'
import { Canvas, useFrame, useThree, ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useSceneStore, useAnimationStore, useUIStore, useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import { generateManifoldLandscape, LandscapeData } from '../../utils/landscapes'
import styles from './Canvas3D.module.css'

// Height scale factor for the landscape
const HEIGHT_SCALE = 2.0

interface LandscapeMeshProps {
  data: LandscapeData
}

function LandscapeMesh({ data, meshRef }: LandscapeMeshProps & { meshRef?: React.RefObject<THREE.Mesh> }) {
  const theme = useUIStore(state => state.theme)
  const pickingMode = useUIStore(state => state.pickingMode)
  const setPickingMode = useUIStore(state => state.setPickingMode)
  const setStartPosition = useSceneStore(state => state.setStartPosition)
  const manifoldRange = useSceneStore(state => state.manifoldRange)
  const animationState = useAnimationStore(state => state.state)
  
  const handleMeshClick = useCallback((event: ThreeEvent<MouseEvent>) => {
    if (!pickingMode || animationState === 'playing') return
    
    event.stopPropagation()
    
    // Get the intersection point in world coordinates
    const point = event.point
    const worldX = point.x
    const worldZ = point.z
    
    // Convert world coordinates back to parameter space
    // Inverse of: worldX = ((paramX - rangeMin) / rangeSize - 0.5) * 10
    const [rangeMin, rangeMax] = manifoldRange
    const rangeSize = rangeMax - rangeMin
    
    const paramX = ((worldX / 10) + 0.5) * rangeSize + rangeMin
    const paramY = ((worldZ / 10) + 0.5) * rangeSize + rangeMin
    
    // Clamp to manifold range
    const clampedX = Math.max(rangeMin, Math.min(rangeMax, paramX))
    const clampedY = Math.max(rangeMin, Math.min(rangeMax, paramY))
    
    // Update start position
    setStartPosition(clampedX, clampedY)
    
    // Exit picking mode
    setPickingMode(false)
  }, [pickingMode, animationState, manifoldRange, setStartPosition, setPickingMode])
  
  const geometry = useMemo(() => {
    const rows = data.z.length
    const cols = data.z[0].length
    
    const geo = new THREE.PlaneGeometry(10, 10, cols - 1, rows - 1)
    const positions = geo.attributes.position
    
    // Find z range
    const zFlat = data.z.flat()
    const zMin = Math.min(...zFlat)
    const zMax = Math.max(...zFlat)
    const zRangeVal = zMax - zMin
    
    // Update vertex positions
    for (let i = 0; i < positions.count; i++) {
      const row = Math.floor(i / cols)
      const col = i % cols
      
      if (row < rows && col < cols) {
        const normalizedZ = (data.z[row][col] - zMin) / zRangeVal
        positions.setZ(i, normalizedZ * HEIGHT_SCALE)
        
        const xPos = (col / (cols - 1) - 0.5) * 10
        const yPos = (row / (rows - 1) - 0.5) * 10
        positions.setX(i, xPos)
        positions.setY(i, yPos)
      }
    }
    
    positions.needsUpdate = true
    geo.computeVertexNormals()
    
    // Create vertex colors
    const colorArray: number[] = []
    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i)
      const normalizedZ = z / HEIGHT_SCALE
      
      let r: number, g: number, b: number
      if (normalizedZ < 0.25) {
        r = 0; g = normalizedZ * 4; b = 1
      } else if (normalizedZ < 0.5) {
        r = 0; g = 1; b = 1 - (normalizedZ - 0.25) * 4
      } else if (normalizedZ < 0.75) {
        r = (normalizedZ - 0.5) * 4; g = 1; b = 0
      } else {
        r = 1; g = 1 - (normalizedZ - 0.75) * 4; b = 0
      }
      
      colorArray.push(r, g, b)
    }
    
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3))
    
    return geo
  }, [data])
  
  const wireframeColor = theme === 'dark' ? '#444444' : '#999999'
  
  return (
    <>
      <mesh 
        ref={meshRef}
        geometry={geometry} 
        rotation={[-Math.PI / 2, 0, 0]}
        onClick={handleMeshClick}
      >
        <meshPhongMaterial vertexColors side={THREE.DoubleSide} flatShading={false} />
      </mesh>
      <mesh geometry={geometry.clone()} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <meshBasicMaterial color={wireframeColor} wireframe transparent opacity={0.15} depthWrite={false} />
      </mesh>
    </>
  )
}

interface StartPointMarkerProps {
  x: number
  y: number
  landscapeData: LandscapeData | null
}

function StartPointMarker({ x, y, landscapeData }: StartPointMarkerProps) {
  if (!landscapeData) return null
  
  const [rangeMin, rangeMax] = landscapeData.x_range
  const rangeSize = rangeMax - rangeMin
  
  // Map parameter space to world space
  const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10
  const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10
  
  // Calculate height at this position
  const normalizedX = (x - rangeMin) / rangeSize
  const normalizedY = (y - rangeMin) / rangeSize
  
  const resolution = landscapeData.z.length
  const col = Math.min(Math.floor(normalizedX * (resolution - 1)), resolution - 2)
  const row = Math.min(Math.floor(normalizedY * (resolution - 1)), resolution - 2)
  
  const zFlat = landscapeData.z.flat()
  const zMin = Math.min(...zFlat)
  const zMax = Math.max(...zFlat)
  const zRange = zMax - zMin
  
  const z00 = (landscapeData.z[row][col] - zMin) / zRange * HEIGHT_SCALE
  const z01 = (landscapeData.z[row][Math.min(col + 1, resolution - 1)] - zMin) / zRange * HEIGHT_SCALE
  const z10 = (landscapeData.z[Math.min(row + 1, resolution - 1)][col] - zMin) / zRange * HEIGHT_SCALE
  const z11 = (landscapeData.z[Math.min(row + 1, resolution - 1)][Math.min(col + 1, resolution - 1)] - zMin) / zRange * HEIGHT_SCALE
  
  const fx = normalizedX * (resolution - 1) - col
  const fy = normalizedY * (resolution - 1) - row
  
  const h0 = z00 * (1 - fx) + z01 * fx
  const h1 = z10 * (1 - fx) + z11 * fx
  const worldY = h0 * (1 - fy) + h1 * fy
  
  const points: [number, number, number][] = [
    [worldX, worldY, worldZ],
    [worldX, worldY + 1.5, worldZ],
  ]
  
  return (
    <Line
      points={points}
      color="#ffaa00"
      lineWidth={3}
      transparent
      opacity={0.9}
    />
  )
}

interface TrajectoryLinesProps {
  trajectories: Record<string, Array<{ x: number; y: number; z: number }>>
  landscapeData: LandscapeData | null
  currentStep: number
  showTrails: boolean
}

function TrajectoryLines({ trajectories, landscapeData, currentStep, showTrails }: TrajectoryLinesProps) {
  if (!landscapeData || !showTrails || !trajectories || Object.keys(trajectories).length === 0) return null
  
  const [rangeMin, rangeMax] = landscapeData.x_range
  const rangeSize = rangeMax - rangeMin
  
  const zFlat = landscapeData.z.flat()
  const zMin = Math.min(...zFlat)
  const zMax = Math.max(...zFlat)
  const zRange = zMax - zMin || 1 // Prevent division by zero
  
  return (
    <>
      {Object.entries(trajectories).map(([optimizer, points]) => {
        if (!points || points.length === 0) return null
        
        // Show at least up to currentStep + 1, minimum 2 points for a line
        const endIndex = Math.max(2, currentStep + 1)
        const visiblePoints = points.slice(0, Math.min(endIndex, points.length))
        if (visiblePoints.length < 2) return null
        
        const linePoints: [number, number, number][] = visiblePoints.map(p => {
          const worldX = ((p.x - rangeMin) / rangeSize - 0.5) * 10
          const worldZ = ((p.y - rangeMin) / rangeSize - 0.5) * 10
          const worldY = ((p.z - zMin) / zRange) * HEIGHT_SCALE + 0.05
          return [worldX, worldY, worldZ]
        })
        
        const color = OPTIMIZER_COLORS[optimizer] || '#ffffff'
        
        return (
          <Line
            key={optimizer}
            points={linePoints}
            color={color}
            lineWidth={2}
          />
        )
      })}
    </>
  )
}

function OptimizerBalls({ trajectories, landscapeData, currentStep }: Omit<TrajectoryLinesProps, 'showTrails'>) {
  const enabled = useOptimizerStore(state => state.enabled)
  
  if (!landscapeData || !trajectories || Object.keys(trajectories).length === 0) return null
  
  const [rangeMin, rangeMax] = landscapeData.x_range
  const rangeSize = rangeMax - rangeMin
  
  const zFlat = landscapeData.z.flat()
  const zMin = Math.min(...zFlat)
  const zMax = Math.max(...zFlat)
  const zRange = zMax - zMin
  
  return (
    <>
      {Object.entries(trajectories).map(([optimizer, points]) => {
        if (!points || points.length === 0 || !enabled[optimizer as keyof typeof enabled]) return null
        
        const pointIndex = Math.min(currentStep, points.length - 1)
        const p = points[pointIndex]
        
        const worldX = ((p.x - rangeMin) / rangeSize - 0.5) * 10
        const worldZ = ((p.y - rangeMin) / rangeSize - 0.5) * 10
        const worldY = ((p.z - zMin) / zRange) * HEIGHT_SCALE + 0.1
        
        const color = OPTIMIZER_COLORS[optimizer] || '#ffffff'
        
        return (
          <mesh key={optimizer} position={[worldX, worldY, worldZ]}>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
          </mesh>
        )
      })}
    </>
  )
}

// Animation controller component that runs inside the Canvas
function AnimationController() {
  const animationState = useAnimationStore(state => state.state)
  const currentStep = useAnimationStore(state => state.currentStep)
  const totalSteps = useAnimationStore(state => state.totalSteps)
  const speed = useAnimationStore(state => state.speed)
  const setCurrentStep = useAnimationStore(state => state.setCurrentStep)
  const pause = useAnimationStore(state => state.pause)
  
  const accumulatorRef = useRef(0)
  
  useFrame((_, delta) => {
    if (animationState !== 'playing') {
      accumulatorRef.current = 0
      return
    }
    
    // Accumulate time, advance step based on speed
    // Base rate: ~100 steps per second at speed 1.0 (fast enough for 10k+ step trajectories)
    accumulatorRef.current += delta * speed * 100
    
    if (accumulatorRef.current >= 1) {
      const stepsToAdvance = Math.floor(accumulatorRef.current)
      accumulatorRef.current -= stepsToAdvance
      
      const newStep = currentStep + stepsToAdvance
      if (newStep >= totalSteps) {
        setCurrentStep(totalSteps - 1)
        pause()
      } else {
        setCurrentStep(newStep)
      }
    }
  })
  
  return null
}

// Click handler for picking points on the landscape
function ClickHandler({ landscapeData, landscapeRef }: { landscapeData: LandscapeData, landscapeRef: React.RefObject<THREE.Mesh> }) {
  const pickingMode = useUIStore(state => state.pickingMode)
  const setPickingMode = useUIStore(state => state.setPickingMode)
  const setStartPosition = useSceneStore(state => state.setStartPosition)
  const manifoldRange = useSceneStore(state => state.manifoldRange)
  const animationState = useAnimationStore(state => state.state)
  
  const handleMeshClick = useCallback((event: ThreeEvent<MouseEvent>) => {
    if (!pickingMode || animationState === 'playing') return
    
    event.stopPropagation()
    
    // Get the intersection point in world coordinates
    const point = event.point
    const worldX = point.x
    const worldZ = point.z
    
    // Convert world coordinates back to parameter space
    // Inverse of: worldX = ((paramX - rangeMin) / rangeSize - 0.5) * 10
    const [rangeMin, rangeMax] = manifoldRange
    const rangeSize = rangeMax - rangeMin
    
    const paramX = ((worldX / 10) + 0.5) * rangeSize + rangeMin
    const paramY = ((worldZ / 10) + 0.5) * rangeSize + rangeMin
    
    // Clamp to manifold range
    const clampedX = Math.max(rangeMin, Math.min(rangeMax, paramX))
    const clampedY = Math.max(rangeMin, Math.min(rangeMax, paramY))
    
    // Update start position
    setStartPosition(clampedX, clampedY)
    
    // Exit picking mode
    setPickingMode(false)
  }, [pickingMode, animationState, manifoldRange, setStartPosition, setPickingMode])
  
  return null
}

function Scene() {
  const theme = useUIStore(state => state.theme)
  const currentManifoldId = useSceneStore(state => state.currentManifoldId)
  const manifoldParams = useSceneStore(state => state.manifoldParams)
  const manifoldRange = useSceneStore(state => state.manifoldRange)
  const startX = useSceneStore(state => state.startX)
  const startY = useSceneStore(state => state.startY)
  const trajectories = useSceneStore(state => state.trajectories)
  const currentStep = useAnimationStore(state => state.currentStep)
  const showTrails = useAnimationStore(state => state.showTrails)
  
  const landscapeMeshRef = useRef<THREE.Mesh>(null)
  
  const landscapeData = useMemo(() => {
    return generateManifoldLandscape(
      currentManifoldId,
      80,
      manifoldRange,
      manifoldRange,
      manifoldParams
    )
  }, [currentManifoldId, manifoldParams, manifoldRange])
  
  const bgColor = theme === 'dark' ? '#0a0a0a' : '#fafafa'
  const gridColor1 = theme === 'dark' ? '#444444' : '#cccccc'
  const gridColor2 = theme === 'dark' ? '#222222' : '#e8e8e8'
  
  return (
    <>
      <color attach="background" args={[bgColor]} />
      
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <directionalLight position={[-10, -10, -5]} intensity={0.4} />
      
      {/* Grid */}
      <gridHelper args={[20, 20, gridColor1, gridColor2]} />
      
      {/* Axes */}
      <axesHelper args={[5]} />
      
      {/* Landscape */}
      <LandscapeMesh data={landscapeData} meshRef={landscapeMeshRef} />
      
      {/* Start point marker */}
      <StartPointMarker x={startX} y={startY} landscapeData={landscapeData} />
      
      {/* Trajectory lines */}
      <TrajectoryLines 
        trajectories={trajectories} 
        landscapeData={landscapeData}
        currentStep={currentStep}
        showTrails={showTrails}
      />
      
      {/* Optimizer balls */}
      <OptimizerBalls 
        trajectories={trajectories}
        landscapeData={landscapeData}
        currentStep={currentStep}
      />
      
      {/* Camera controls */}
      <OrbitControls 
        enableDamping 
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
      />
      
      {/* Animation controller */}
      <AnimationController />
    </>
  )
}

export function Canvas3D() {
  const pickingMode = useUIStore(state => state.pickingMode)
  
  return (
    <div 
      className={`${styles.canvasContainer} ${pickingMode ? styles.pickingMode : ''}`}
    >
      <Canvas
        camera={{ position: [15, 15, 15], fov: 60 }}
        gl={{ antialias: true }}
      >
        <Scene />
      </Canvas>
    </div>
  )
}
