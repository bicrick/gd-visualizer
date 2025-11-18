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
