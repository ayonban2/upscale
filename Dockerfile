# Use an official CUDA runtime image with Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1-mesa-glx && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy your application code and models
COPY . .

# Expose port used by FastAPI
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
