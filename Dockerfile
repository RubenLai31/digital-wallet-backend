# syntax=docker/dockerfile:1

# --- Build stage: install dependencies into a virtualenv ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install deps first (layer caching — this only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage: slim image with only what's needed to run ---
FROM python:3.12-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the application code
COPY . .

# Fly.io sets $PORT automatically — uvicorn reads it from the env
# --host 0.0.0.0: listen on all interfaces (required inside a container)
# --workers 1: single worker (fine for free tier; bump to 2-4 on paid plans)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
