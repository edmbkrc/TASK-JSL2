# Base image
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpoppler-cpp-dev \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Expose Ollama port
EXPOSE 11434

# Start Ollama server and the app
CMD ["bash", "-c", "ollama serve & streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]
