FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/models

# Copy server code
COPY server.py .

# Railway requires the app to bind to 0.0.0.0 and use PORT env
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
