FROM python:3.11-slim

# System dependencies for audio/video processing and nilearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Create directories
RUN mkdir -p /app/cache /app/uploads

ENV CACHE_DIR=/app/cache
ENV UPLOAD_DIR=/app/uploads
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOAD_MODEL_ON_STARTUP=true

EXPOSE 8000

CMD ["python", "main.py"]
