# Gradient Descent Visualizer

An interactive 3D visualization tool for comparing gradient descent optimization algorithms. Watch as different optimizers (SGD, Batch GD, and Momentum GD) navigate a loss landscape in real-time.

![Gradient Descent Visualization](GD.png)

## What is this?

This project visualizes how different gradient descent algorithms optimize towards local minima on a 3D loss surface. Each optimizer is represented as a colored ball that "rolls" down the landscape, allowing you to:

- Compare optimization paths side-by-side
- Adjust learning rates, momentum, and iterations in real-time
- See how different starting positions affect convergence
- Visualize trajectory trails to understand optimizer behavior

## Getting Started with Docker

1. **Start the application:**
```bash
docker-compose up --build
```

2. **Open your browser:**
Navigate to `http://localhost:8080`

3. **Stop the application:**
```bash
docker-compose down
```

That's it! The containerized setup handles both frontend and backend, eliminating any CORS issues.

## Usage

Once the application is running:
1. The 3D loss landscape will load automatically
2. Adjust parameters (learning rate, momentum, iterations) using the control panel
3. Set a starting position or use "Random Start"
4. Click "Play" to watch the optimizers in action
5. Use Play/Pause/Reset to control the animation

## Tech Stack

- **Backend**: Python/Flask with NumPy for optimization computations
- **Frontend**: JavaScript with Three.js for 3D rendering

## Optimization Algorithms

This visualizer includes four gradient descent variants:

- **Stochastic Gradient Descent (SGD)**: Updates parameters using one sample at a time, leading to noisy but fast updates. Note: SGD is simulated in this visualization to demonstrate its characteristic noisy behavior.

- **Batch Gradient Descent**: Computes gradients using the entire dataset, resulting in smooth, stable trajectories but slower convergence per step.

- **Momentum Gradient Descent**: Accelerates convergence by accumulating a velocity vector in directions of persistent reduction, helping navigate ravines and flat regions more effectively.

- **ADAM (Adaptive Moment Estimation)**: Combines momentum with adaptive learning rates for each parameter, automatically adjusting step sizes based on gradient history for efficient and robust optimization.
