/**
 * API utilities for communicating with the backend
 */

const API_BASE_URL = import.meta.env.DEV 
  ? 'http://localhost:5001/api'
  : 'https://gd-experiments-1031734458893.us-central1.run.app/api'

export interface ManifoldResponse {
  manifolds: Array<{
    id: string
    name: string
    description: string
    default_range: [number, number]
    parameters?: Array<{
      name: string
      label: string
      min: number
      max: number
      step: number
      default: number
    }>
  }>
}

export interface OptimizationRequest {
  manifold: string
  initial_params: [number, number]
  learning_rate: number
  momentum: number
  n_iterations: number
  seed: number
  use_convergence: boolean
  max_iterations: number
  convergence_threshold: number
  enabled_optimizers: Record<string, boolean>
  optimizer_params: Record<string, unknown>
  manifold_params: Record<string, number>
}

export interface TrajectoryPoint {
  x: number
  y: number
  z: number
}

// Backend returns trajectories as arrays of [x, y, z] tuples
export type TrajectoryTuple = [number, number, number]

export interface OptimizationResponse {
  batch?: TrajectoryTuple[]
  momentum?: TrajectoryTuple[]
  adam?: TrajectoryTuple[]
  sgd?: TrajectoryTuple[]
  wheel?: TrajectoryTuple[]
  manifold_id?: string
}

/**
 * Fetch available manifolds from the backend
 */
export async function fetchManifolds(): Promise<ManifoldResponse> {
  const response = await fetch(`${API_BASE_URL}/manifolds`, {
    cache: 'no-store'
  })
  if (!response.ok) {
    throw new Error(`Failed to fetch manifolds: ${response.status}`)
  }
  return response.json()
}

/**
 * Run optimization and get trajectories
 */
export async function runOptimization(params: OptimizationRequest): Promise<OptimizationResponse> {
  const response = await fetch(`${API_BASE_URL}/optimize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
  
  if (!response.ok) {
    throw new Error(`Optimization failed: ${response.status}`)
  }
  
  return response.json()
}
