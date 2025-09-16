# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/plots static/sample_images

# Generate sample images at build time
RUN python generate_samples.py && echo "Sample images generated successfully" || echo "Sample generation failed, will try at runtime"

# Expose port
EXPOSE $PORT

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check to verify app is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:' + str(os.environ.get('PORT', 5000)) + '/health')" || exit 1

# Run the application
CMD ["python", "app.py"]