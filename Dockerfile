# Use CUDA base with Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirement file first (layer caching)
COPY requirements.txt .

# Install Python packages without cache to reduce size
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

# Copy all code files (excluding models and junk via `.dockerignore`)
COPY . .

# Create temp and models directories
RUN mkdir -p temp models

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
