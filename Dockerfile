FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY best_face_embedding_model.pth ./

# Create necessary directories
RUN mkdir -p artifacts/outputs identities

# Copy pre-trained model and config if exists
COPY artifacts/outputs/employee_db.pt ./artifacts/outputs/ 2>/dev/null || true
COPY artifacts/outputs/gui_threshold.json ./artifacts/outputs/ 2>/dev/null || true

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
