import os
import io
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from google.cloud import storage
import threading
import subprocess

app = Flask(__name__)

# --- 1. CONSOLIDATED HELPERS ---
def get_gcs_resource():
    """Single source of truth for GCS access."""
    bucket_name = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
    client = storage.Client()
    return client, client.bucket(bucket_name)

def get_collective_memory_data():
    """Shared logic for both the dashboard and the trainer."""
    _, bucket = get_gcs_resource()
    blobs = bucket.list_blobs(prefix="uploads/")
    
    file_list = []
    for blob in blobs:
        if blob.name.endswith(".csv"):
            file_list.append({
                "name": blob.name.replace("uploads/", ""),
                "full_path": blob.name,
                "size": f"{round(blob.size / 1024, 2)} KB",
                "updated": blob.updated.strftime('%Y-%m-%d %H:%M:%S'),
                "type": "Structured CSV"
            })
    file_list.sort(key=lambda x: x['updated'], reverse=True)
    return file_list, bucket.name

# --- 2. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        memories, bucket_name = get_collective_memory_data()
        return render_template('dashboard.html', files=memories, bucket_name=bucket_name, view_mode="Collective Memory")
    except Exception as e:
        return f"Error loading Collective Memory: {e}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"status": "error", "message": "No file part"}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {"status": "error", "message": "No selected file"}, 400

    try:
        # 1. Read the file into memory
        file_bytes = file.read()
        
        # 2. Upload directly to GCS
        # This will automatically trigger your Lambda/Background process
        upload_to_gcs(file_bytes, file.filename, file.content_type)
        
        # 3. Respond immediately
        return {
            "status": "success",
            "message": f"'{file.filename}' added to Collective Memory. Training started in background.",
            "redirect_hint": url_for('dashboard')
        }, 202  # 202 means 'Accepted' for processing
        
    except Exception as e:
        app.logger.error(f"Upload failed: {e}")
        return {"status": "error", "message": "Failed to store memory."}, 500

@app.route('/predict', methods=['GET'])
def predict():
    sqft = request.args.get('sqft')
    if not sqft:
        return {"error": "Missing 'sqft' parameter"}, 400

    try:
        # 1. Get the latest model from GCS (Collective Memory)
        # This replaces the local os.path.exists('model.pkl') check
        _, bucket = get_gcs_resource()
        blob = bucket.blob("models/model.pkl") # Assuming Lambda saves here
        
        if not blob.exists():
            return {"error": "Model is still being trained in the cloud. Please wait."}, 503

        # 2. Download model into memory (don't save to disk to keep it lean)
        model_bytes = blob.download_as_bytes()
        model = pickle.loads(model_bytes)
        
        # 3. Perform prediction
        # We still need to cast to float for the model
        prediction = model.predict([[float(sqft)]])
        result = "EXPENSIVE" if prediction[0] == 1 else "AFFORDABLE"
        
        return {
            "sqft": sqft, 
            "prediction": result,
            "source": "Cloud Memory Model"
        }
        
    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return {"error": "Failed to process prediction. Ensure data is formatted correctly."}, 500

@app.route('/trigger-train', methods=['POST'])
def trigger_train():
    """
    Triggers the separate train.py script.
    Using a thread prevents the web UI from freezing during training.
    """
    def run_training_script():
        try:
            app.logger.info("Background training started...")
            # This calls your separate train.py file
            result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            if result.returncode == 0:
                app.logger.info("Background training completed successfully.")
            else:
                app.logger.error(f"Training script failed: {result.stderr}")
        except Exception as e:
            app.logger.error(f"Error during background training: {e}")

    # Fire and forget
    thread = threading.Thread(target=run_training_script)
    thread.start()

    return {"status": "success", "message": "Training initiated in the cloud."}, 202

# --- 3. TO BE MOVED TO LAMBDA/CONSOLE ---
# train_with_memory()
# clean_and_validate()
# These functions should live in 'trainer.py' or 'console_interface.py'

if __name__ == "__main__":
    # 0.0.0.0 is MANDATORY for Docker
    # port 8080 must match the internal part of your -p 5005:8080
    app.run(host="0.0.0.0", port=8080)