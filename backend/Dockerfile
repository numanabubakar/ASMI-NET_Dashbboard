# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY backend/ ./backend/
COPY model/ ./model/

# Set Python path to include backend directory
ENV PYTHONPATH="/app/backend:${PYTHONPATH}"

# Expose port (HF Spaces use 7860)
EXPOSE 7860

# Run the application
CMD ["python", "backend/app.py"]
