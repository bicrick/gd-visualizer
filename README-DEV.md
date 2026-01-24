# Development Setup

## Quick Start (Local Development)

Start everything locally for fast iteration:

```bash
npm run dev
```

This runs both:
- **Backend**: http://localhost:5001 (Flask API) 
- **Frontend**: http://localhost:3000 (Vite dev server with hot reload)

Frontend proxies `/api` requests to the backend automatically.

## Docker Development (Matches Production)

To test in Docker containers (matches Cloud Run deployment):

```bash
npm run dev:docker
```

This runs both backend and frontend in Docker containers:
- Backend: http://localhost:5001 (Flask API in Python container)
- Frontend: http://localhost:3000 (Vite dev server in Node container)

Stop containers:
```bash
npm run stop
```

## Manual Setup

### Backend Only
```bash
cd backend
pip3 install -r requirements.txt
PORT=5001 python3 app.py
```

### Frontend Only  
```bash
cd frontend
npm install
npm run dev
```

## Production Build Test

Test the production build locally:

```bash
cd frontend
npm run build
npm run preview
```

## Deployment

- **Backend**: Deploys to Cloud Run via Docker
  ```bash
  npm run deploy:backend
  ```

- **Frontend**: Deploys to Vercel
  ```bash
  npm run deploy:frontend
  ```
