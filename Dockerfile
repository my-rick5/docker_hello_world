FROM python:3.9-slim

# Keep the OS lean
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PERSISTENCE SETUP ---
USER root
# Create MLflow DB directory and the log file ahead of time
RUN mkdir -p /var/lib/mlflow && \
    touch /app/training.log && \
    chmod -R 777 /var/lib/mlflow && \
    chmod 777 /app/training.log

# Copy only what's needed for the app
COPY app.py .
COPY train.py .
COPY templates/ ./templates/

# Ensure the app runs as a non-root user for security (optional but recommended)
# For now, we stay as root to ensure the volume mounts don't have permission fits
EXPOSE 8080
CMD ["python", "app.py"]