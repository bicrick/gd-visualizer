FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY backend code (no frontend)
COPY backend/*.py ./backend/

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Set environment variable for production
ENV PYTHONUNBUFFERED=1

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 300 --chdir /app/backend app:app


