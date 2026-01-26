# Docker Development Setup

This project supports Docker-based development with separate containers for frontend and backend.

## Quick Start

From the project root, start both frontend and backend in Docker containers:

```bash
npm run dev
```

This will:
- Build and start the backend container (Flask) on port 5001
- Build and start the frontend container (Vite/React) on port 3000
- Enable hot-reload for both services
- Create a Docker network for inter-container communication

## Accessing the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:5001/api/health

## Useful Commands

### View logs from both services
```bash
npm run dev:logs
```

### View logs from specific service
```bash
npm run dev:logs:backend
npm run dev:logs:frontend
```

### Stop all containers
```bash
npm run dev:stop
```

### Check running containers
```bash
docker ps
```

### Run locally without Docker (old method)
```bash
npm run dev:local
```

## Container Details

### Backend Container
- **Name**: `gd-backend`
- **Port**: 5001
- **Hot Reload**: Enabled (volume mounted)
- **Image**: Built from `docker/Dockerfile.backend`

### Frontend Container
- **Name**: `gd-frontend`
- **Port**: 3000
- **Hot Reload**: Enabled (volume mounted)
- **Image**: Built from `docker/Dockerfile.frontend`
- **API Proxy**: Configured to proxy `/api` requests to backend container

## Debugging

You can attach to running containers for debugging:

```bash
# Backend
docker exec -it gd-backend /bin/bash

# Frontend
docker exec -it gd-frontend /bin/sh
```

## Troubleshooting

### Containers won't start or module errors (ARM64/M1 Mac issues)
If you see errors about missing native modules like `@rollup/rollup-linux-arm64-gnu`, clean up everything and rebuild:
```bash
npm run dev:clean
npm run dev
```

This removes all volumes and containers, forcing a fresh install of dependencies inside the container.

### Port conflicts
Make sure ports 3000 and 5001 are not in use by other applications.

### Changes not reflecting
The volumes are mounted for hot reload, but if you're having issues:
```bash
npm run dev:stop
npm run dev
```
