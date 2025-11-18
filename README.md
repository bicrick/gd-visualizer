# Gradient Descent Visualizer

An interactive 3D visualization tool for comparing different gradient descent optimization algorithms. Watch animated "balls" roll down a loss landscape as they optimize, comparing Stochastic Gradient Descent, Batch Gradient Descent, and Momentum Gradient Descent side-by-side.

## Features

- **3D Loss Landscape**: Beautifully rendered 3D mesh showing the loss function surface
- **Multiple Optimizers**: Compare SGD, Batch GD, and Momentum GD simultaneously
- **Interactive Controls**: Adjust learning rate, momentum, iterations, and starting position
- **Animated Visualization**: Watch optimizers as colored balls rolling down the landscape
- **Trajectory Trails**: See the exact path each optimizer takes
- **Local Minima Demo**: Different starting positions lead to different final minima

## Quick Start with Docker (Recommended)

The easiest way to run the application is using Docker Compose. This eliminates CORS issues and ensures consistent setup.

1. **Build and start the container:**
```bash
docker-compose up --build
```

2. **Open your browser:**
Navigate to `http://localhost:8080`

The application will be fully functional - both frontend and backend are served from the same container, eliminating CORS issues.

To stop the container:
```bash
docker-compose down
```

## Manual Setup

### Backend

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

**Important**: Make sure the backend is running before opening the frontend. You should see output like:
```
 * Running on http://127.0.0.1:5000
```

### Frontend

1. Open `frontend/index.html` in a web browser, or serve it using a local web server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8000
```

Then navigate to `http://localhost:8000`

## Usage

1. **Load the Landscape**: The loss landscape will automatically load when you open the page
2. **Set Parameters**: Adjust learning rate, momentum, and number of iterations using the sliders
3. **Choose Starting Position**: Set initial X and Y coordinates, or click "Random Start"
4. **Run Optimization**: Click "Play" to compute and visualize the optimization trajectories
5. **Control Animation**: Use Play/Pause/Reset to control the animation, and adjust speed
6. **Toggle Trails**: Check/uncheck "Show Trajectory Trails" to toggle path visualization

## Architecture

- **Backend** (Python/Flask): Computes optimization trajectories using NumPy
- **Frontend** (JavaScript/Three.js): Renders 3D visualization and handles user interactions

## File Structure

```
backend/
  - app.py              # Flask API server
  - loss_functions.py   # Synthetic loss landscapes
  - optimizers.py       # SGD, Batch, Momentum implementations
  - synthetic_data.py   # Training data generation
  - requirements.txt    # Python dependencies

frontend/
  - index.html         # Main HTML file
  - css/style.css      # Styling
  - js/
    - scene.js         # Three.js scene setup
    - optimizers.js    # Ball animations and trajectories
    - controls.js      # UI event handlers
```

## Troubleshooting

### CORS Errors
If you see CORS errors in the browser console:
1. Make sure the backend is running (`python app.py` in the `backend` directory)
2. Verify the backend is accessible at `http://localhost:5000/api/health`
3. Restart the backend server if you just updated the code

### Backend Not Starting
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 5000 is not already in use
- Verify Python version (3.7+ recommended)

### Frontend Not Loading Landscape
- Check browser console for errors
- Verify backend is running and accessible
- Try accessing `http://localhost:5000/api/health` directly in your browser

## Future Enhancements

- Additional optimizers (Adam, RMSprop, AdaGrad, Nesterov)
- Multiple loss function presets
- 2D contour view alongside 3D view
- Export trajectory data or animations

