# Gradient Descent Experiments

Interactive 3D visualization of gradient descent optimization algorithms with various loss landscapes.

## Architecture

- **Frontend**: Static HTML/CSS/JS hosted on Vercel
- **Backend**: Flask API on Google Cloud Run (scales to zero)
- **Cost**: ~$0-2/month with light usage

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for npm scripts)
- gcloud CLI (authenticated as patrickbrownai@gmail.com)
- Vercel CLI

### Local Development

```bash
# Install dependencies
npm install
pip install -r backend/requirements.txt

# Run locally (backend on :5000, frontend on :3000)
npm run dev
```

Visit http://localhost:3000 to see the frontend.

### Deployment

```bash
# Deploy both frontend and backend
npm run deploy

# Or deploy individually
npm run deploy:backend   # Deploy to Cloud Run
npm run deploy:frontend  # Deploy to Vercel
```

### Useful Commands

```bash
# Check deployment status
npm run status

# Test endpoints
npm run test:backend
npm run test:frontend

# View backend logs
npm run logs:backend
```

## Live URLs

- **Production**: https://gd.bicrick.com
- **Vercel**: https://gd-visualizer.vercel.app
- **Backend API**: https://gd-experiments-1031734458893.us-central1.run.app/api

## Features

- Multiple loss landscapes with adjustable parameters
- 6 optimization algorithms (SGD, Batch GD, Momentum, Adam, Ballistic, Ballistic Adam)
- Real-time 3D visualization with Three.js
- Interactive parameter tuning
- Dark/Light theme toggle
- Rate limiting: 50 requests/hour per IP

## Project Structure

```
.
├── backend/                 # Flask API
│   ├── app.py              # Main API server
│   ├── loss_functions.py   # Loss landscape definitions
│   ├── optimizers.py       # Optimization algorithms
│   └── requirements.txt    # Python dependencies
├── frontend/               # Static site
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── vercel.json        # Vercel configuration
├── Dockerfile             # Backend container
├── .gcloudignore         # Cloud Run deployment exclusions
└── package.json          # Deployment scripts
```

## Cost Protection

1. **Cloud Run**: max-instances=1, scales to zero when idle
2. **Rate Limiting**: 50 requests/hour per IP
3. **GCP Billing Alerts**: $5, $10, $20 thresholds

## Local Development Setup

### Backend Configuration

When running locally, the backend uses PORT 5000. The frontend is configured to point to the production Cloud Run URL by default. To use local backend:

1. Update `frontend/js/scene.js` line 18:
```javascript
window.API_BASE_URL = 'http://localhost:5000/api';
```

2. Run backend: `python backend/app.py`

3. Serve frontend: `npx serve frontend -l 3000`

## Technologies

- **Frontend**: Three.js, Vanilla JavaScript
- **Backend**: Flask, NumPy, PyTorch
- **Hosting**: Vercel (frontend), Google Cloud Run (backend)
- **Domain**: Squarespace DNS → Vercel

## License

MIT
