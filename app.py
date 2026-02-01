import os
import io
import threading
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from google.cloud import storage
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# --- INITIALIZE CLIENT GLOBALLY ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
storage_client = storage.Client()

def get_gcs_resource():
    """Helper to initialize the GCS client."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

# ---- Model History ---

def get_model_history():
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model_name = "HousingPriceModel"
    
    history = []
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            run = client.get_run(v.run_id)
            accuracy = run.data.metrics.get("accuracy", "N/A")
            
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "accuracy": round(accuracy, 4) if isinstance(accuracy, float) else accuracy,
                "created": v.creation_timestamp
            })
    except Exception as e:
        print(f"MLflow Fetch Error: {e}")
        
    return sorted(history, key=lambda x: int(x['version']), reverse=True)

# --- Routes ---

@app.route('/')
def root():
    """The 'Longview' Fix: Redirect base URL directly to dashboard."""
    return redirect(url_for('dashboard'))

@app.route('/models')
def index():
    """Main page with Predict form (formerly the index)."""
    model_data = [
        {"version": "1", "stage": "Production", "timestamp": "2024-05-20"},
        {"version": "2", "stage": "Staging", "timestamp": "2024-05-21"},
    ]
    return render_template('index.html', models=model_data)

@app.route('/dashboard')
def dashboard():
    """The Primary Command Center."""
    try:
        _, bucket = get_gcs_resource()
        blobs = bucket.list_blobs() 
        
        files = []
        for blob in blobs:
            files.append({
                'name': blob.name,
                'size': f"{blob.size / 1024:.2f} KB",
                'time': blob.updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        model_versions = get_model_history()
        return render_template('dashboard.html', files=files, model_versions=model_versions)
        
    except Exception as e:
        print(f"❌ Dashboard Error: {e}")
        return f"Error: {e}", 500

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "No file provided", 400
    
    try:
        _, bucket = get_gcs_resource()
        blob = bucket.blob(file.filename) 
        blob.upload_from_file(file)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Upload failed: {e}", 500

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
    try:
        _, bucket = get_gcs_resource()
        blob = bucket.blob(filename)
        if blob.exists():
            blob.delete()
            return redirect(url_for('dashboard'))
        return f"File {filename} not found", 404
    except Exception as e:
        return f"Error deleting file: {e}", 500

@app.route('/trigger-train', methods=['POST'])
def trigger_train():
    def run_training():
        try:
            print("Background training started...")
            subprocess.run(["python", "./train.py"])
            print("Background training completed.")
        except Exception as e:
            print(f"Background training failed: {e}")

    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({"status": "success", "message": "Training started."}), 202

@app.route('/predict', methods=['GET'])
def predict():
    sqft = request.args.get('sqft')
    return f"Prediction for {sqft} sqft: (Mock: EXPENSIVE)"

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        if not blob.exists():
            return "File not found", 404
        file_data = io.BytesIO()
        blob.download_to_file(file_data)
        file_data.seek(0)
        return send_file(file_data, as_attachment=True, download_name=filename.split('/')[-1])
    except Exception as e:
        return f"Internal Error: {e}", 500

@app.route('/promote/<version>', methods=['POST'])
def promote_model(version):
    try:
        client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
        model_name = "HousingPriceModel"
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production", archive_existing_versions=True
        )
        return jsonify({"status": "success", "message": f"v{version} promoted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)