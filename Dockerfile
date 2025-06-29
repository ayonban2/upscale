# ✅ Base: lightweight CPU-only image
FROM python:3.10-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# ✅ Copy requirements first (cache-friendly)
COPY requirements.txt .

# ✅ Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ Copy rest of the app
COPY . .

# ✅ Create folders (not strictly needed but safe)
RUN mkdir -p temp models

# ✅ Expose FastAPI port
EXPOSE 8000

# ✅ Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
