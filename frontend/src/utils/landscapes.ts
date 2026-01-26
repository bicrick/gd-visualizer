/**
 * Landscape function library - client-side landscape generation
 * TypeScript port from landscapes.js
 */

export interface LandscapeData {
  x: number[][]
  y: number[][]
  z: number[][]
  x_range: [number, number]
  y_range: [number, number]
}

export interface ManifoldParams {
  global_scale?: number
  well_width?: number
  well_depth_scale?: number
  num_wells?: number
}

/**
 * Himmelblau's function - has 4 local minima
 */
function himmelblau(x: number, y: number): number {
  return Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2)
}

/**
 * Rastrigin function - highly multimodal with many local minima
 */
function rastrigin(x: number, y: number, A = 10): number {
  return A * 2 + 
    (x * x - A * Math.cos(1.5 * Math.PI * x)) + 
    (y * y - A * Math.cos(1.5 * Math.PI * y))
}

/**
 * Generate well positions in a regular polygon pattern
 */
function generateWellPositions(numWells: number, radius = 4.0, baseDepth = 2.5): [number, number, number][] {
  const wells: [number, number, number][] = []
  const n = Math.floor(numWells)
  
  if (n === 0) return wells
  
  if (n === 1) {
    wells.push([0, 0, baseDepth])
  } else if (n === 2) {
    wells.push([0, radius, baseDepth])
    wells.push([0, -radius, baseDepth])
  } else {
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2
      const wx = radius * Math.cos(angle)
      const wy = radius * Math.sin(angle)
      wells.push([wx, wy, baseDepth])
    }
  }
  
  return wells
}

/**
 * Custom function with multiple local minima (wells/valleys)
 */
function customMultimodal(
  x: number, 
  y: number, 
  globalScale = 0.1, 
  wellWidth = 2.0, 
  wellDepthScale = 1.0, 
  numWells = 6
): number {
  let loss = (x * x + y * y) * globalScale
  
  const wells = generateWellPositions(numWells, 4.0, 2.5)
  
  for (const [wx, wy, depth] of wells) {
    const distSq = (x - wx) * (x - wx) + (y - wy) * (y - wy)
    loss -= depth * wellDepthScale * Math.exp(-distSq / wellWidth)
  }
  
  loss += 15.0
  return loss
}

/**
 * Ackley function - highly multimodal with deep global minimum
 */
function ackley(x: number, y: number): number {
  const a = 20
  const b = 0.2
  const c = 2 * Math.PI
  
  const term1 = -a * Math.exp(-b * Math.sqrt(0.5 * (x * x + y * y)))
  const term2 = -Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)))
  
  return term1 + term2 + a + Math.E
}

type ManifoldFunction = (x: number, y: number, ...args: number[]) => number

const MANIFOLD_FUNCTIONS: Record<string, ManifoldFunction> = {
  'custom_multimodal': customMultimodal,
  'himmelblau': himmelblau,
  'rastrigin': rastrigin,
  'ackley': ackley,
}

/**
 * Generate landscape mesh for a specific manifold
 */
export function generateManifoldLandscape(
  manifoldId: string,
  resolution = 80,
  xRange: [number, number] = [-5, 5],
  yRange: [number, number] = [-5, 5],
  params: ManifoldParams | null = null
): LandscapeData {
  const func = MANIFOLD_FUNCTIONS[manifoldId] || customMultimodal
  
  const [xMin, xMax] = xRange
  const [yMin, yMax] = yRange
  
  const xVals: number[] = []
  const yVals: number[] = []
  
  for (let i = 0; i < resolution; i++) {
    xVals.push(xMin + (xMax - xMin) * i / (resolution - 1))
    yVals.push(yMin + (yMax - yMin) * i / (resolution - 1))
  }
  
  const X: number[][] = []
  const Y: number[][] = []
  const Z: number[][] = []
  
  for (let i = 0; i < resolution; i++) {
    X.push([])
    Y.push([])
    Z.push([])
    
    for (let j = 0; j < resolution; j++) {
      X[i].push(xVals[j])
      Y[i].push(yVals[i])
      
      let zVal: number
      if (params && manifoldId === 'custom_multimodal') {
        zVal = customMultimodal(
          xVals[j], 
          yVals[i],
          params.global_scale ?? 0.1,
          params.well_width ?? 2.0,
          params.well_depth_scale ?? 1.0,
          params.num_wells ?? 6
        )
      } else {
        zVal = func(xVals[j], yVals[i])
      }
      
      Z[i].push(zVal)
    }
  }
  
  return { x: X, y: Y, z: Z, x_range: xRange, y_range: yRange }
}

/**
 * Get z value at a specific (x, y) point for a manifold
 */
export function getManifoldZValue(
  manifoldId: string,
  x: number,
  y: number,
  params: ManifoldParams | null = null
): number {
  const func = MANIFOLD_FUNCTIONS[manifoldId] || customMultimodal
  
  if (params && manifoldId === 'custom_multimodal') {
    return customMultimodal(
      x, y,
      params.global_scale ?? 0.1,
      params.well_width ?? 2.0,
      params.well_depth_scale ?? 1.0,
      params.num_wells ?? 6
    )
  }
  
  return func(x, y)
}
