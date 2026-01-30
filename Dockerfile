FROM python:3.9-slim
RUN pip install flask
WORKDIR /app


RUN mkdir -p data
# Install dependencies first (Cached Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and app (Uncached Layer)
COPY app.py .
COPY templates/ ./templates/


EXPOSE 8080


CMD ["python", "app.py"]
