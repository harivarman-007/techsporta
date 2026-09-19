# ── EmpathAI Backend — Dockerfile ──────────────────────────────────────────
# Base: Python 3.11 slim (3.13 has fewer pre-built wheels for ML deps)
FROM python:3.11-slim

# System deps for OpenCV, audio, and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# torch CPU-only first (GPU requires host CUDA drivers mounted at runtime)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn (respects $PORT from cloud platforms like Railway, defaults to 8000)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"
