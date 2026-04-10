FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for llama-cpp-python compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory (persistent disk on Render)
RUN mkdir -p /app/models

# Copy server code
COPY server.py .

# Render sets PORT automatically, default to 10000
ENV PORT=10000
EXPOSE 10000

CMD ["python", "server.py"]
