FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/*.py ./backend/

# Copy frontend files
COPY frontend /app/frontend

# Set working directory to backend for imports
WORKDIR /app/backend

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]


