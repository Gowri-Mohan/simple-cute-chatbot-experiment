# Base OS
FROM ubuntu:22.04

# Install system dependencies (Python, pip, ffmpeg for audio processing)
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy server code
COPY ./server ./server

# Set environment for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port for Flask
EXPOSE 8000

# Start Flask app
CMD ["python3", "server/app.py"]
