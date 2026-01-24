// Optimizer names
export type OptimizerName = 'sgd' | 'batch' | 'momentum' | 'adam' | 'ballistic' | 'ballistic_adam';

// Animation states
export type AnimationState = 'stopped' | 'playing' | 'paused';

// Theme types
export type Theme = 'light' | 'dark';

// Parameter types
export interface ParameterDefinition {
  id: string;
  label: string;
  type: 'range' | 'checkbox';
  min?: number;
  max?: number;
  step?: number;
  default: number | boolean;
  tooltip?: string;
  conditionalDisplay?: {
    dependsOn: string;
    value: boolean;
  };
}

// Optimizer configuration
export interface OptimizerConfig {
  id: OptimizerName;
  name: string;
  color: string;
  defaultEnabled: boolean;
  params: ParameterDefinition[];
}

// Optimizer parameter values
export interface SGDParams {
  learningRate: number;
  useConvergence: boolean;
  iterations: number;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface BatchParams {
  learningRate: number;
  useConvergence: boolean;
  iterations: number;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface MomentumParams {
  learningRate: number;
  momentum: number;
  lrDecay: number;
  useConvergence: boolean;
  iterations: number;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface AdamParams {
  learningRate: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  useConvergence: boolean;
  iterations: number;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface BallisticParams {
  dropHeight: number;
  gravity: number;
  elasticity: number;
  bounceThreshold: number;
  ballRadius: number;
  maxIterations: number;
}

export interface BallisticAdamParams {
  learningRate: number;
  momentum: number;
  gravity: number;
  dt: number;
  maxAirSteps: number;
  maxBisectionIters: number;
  collisionTol: number;
  useConvergence: boolean;
  iterations: number;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface OptimizerParams {
  sgd: SGDParams;
  batch: BatchParams;
  momentum: MomentumParams;
  adam: AdamParams;
  ballistic: BallisticParams;
  ballistic_adam: BallisticAdamParams;
}

// Trajectory data
export interface TrajectoryPoint {
  x: number;
  y: number;
  loss?: number;
  z?: number;
}

export type Trajectory = TrajectoryPoint[];

export interface OptimizationResults {
  [key: string]: Trajectory;
}

// Manifold types
export interface ManifoldInfo {
  id: string;
  name: string;
  description?: string;
  params?: ManifoldParameterDefinition[];
}

export interface ManifoldParameterDefinition {
  id: string;
  name: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface ManifoldParams {
  [key: string]: number;
}

// API request/response types
export interface OptimizationRequest {
  startX: number;
  startY: number;
  manifoldId: string;
  manifoldParams: ManifoldParams;
  optimizers: {
    [K in OptimizerName]?: OptimizerParams[K] & { enabled: boolean };
  };
}

export interface OptimizationResponse {
  results: OptimizationResults;
}

export interface LandscapeData {
  x: number[];
  y: number[];
  z: number[][];
  zMin: number;
  zMax: number;
}

export interface ClassifierDataPoint {
  x: number;
  y: number;
  label: number;
}

export interface ClassifierDataset {
  points: ClassifierDataPoint[];
}

// App state
export interface AppState {
  // Start position
  startPosition: {
    x: number;
    y: number;
  };
  
  // Animation
  animationState: AnimationState;
  currentStep: number;
  totalSteps: number;
  speed: number;
  
  // Display
  showTrails: boolean;
  pickingMode: boolean;
  
  // Optimizers
  optimizerParams: OptimizerParams;
  enabledOptimizers: Record<OptimizerName, boolean>;
  trajectories: OptimizationResults;
  
  // Manifold
  currentManifoldId: string;
  manifoldParams: ManifoldParams;
  manifolds: ManifoldInfo[];
  
  // Theme
  theme: Theme;
}

// Context actions
export interface AppContextValue extends AppState {
  // Start position actions
  setStartPosition: (x: number, y: number) => void;
  randomizeStartPosition: () => void;
  
  // Animation actions
  setAnimationState: (state: AnimationState) => void;
  setCurrentStep: (step: number) => void;
  setSpeed: (speed: number) => void;
  
  // Display actions
  setShowTrails: (show: boolean) => void;
  setPickingMode: (enabled: boolean) => void;
  
  // Optimizer actions
  updateOptimizerParam: <T extends OptimizerName>(
    optimizer: T,
    param: keyof OptimizerParams[T],
    value: number | boolean
  ) => void;
  toggleOptimizer: (optimizer: OptimizerName, enabled: boolean) => void;
  
  // Manifold actions
  setManifold: (manifoldId: string) => void;
  updateManifoldParam: (param: string, value: number) => void;
  setManifolds: (manifolds: ManifoldInfo[]) => void;
  
  // Optimization actions
  runOptimization: () => Promise<void>;
  clearTrajectories: () => void;
  
  // Theme actions
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}
