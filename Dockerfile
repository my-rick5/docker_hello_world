FROM python:3.9-slim

# Keep the OS lean
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first (this layer stays cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for the app
COPY app.py .
COPY train.py .
COPY templates/ ./templates/

EXPOSE 8080
CMD ["python", "app.py"]