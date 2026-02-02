import os
import io
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from google.cloud import storage
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --- FORCE IN-MEMORY TRACKING ---
# This prevents the "unable to open database file" error on startup
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlflow_persistent.db")
mlflow.set_tracking_uri(TRACKING_URI)

app = Flask(__name__)

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
LOG_FILE_PATH = "/app/training.log"

def get_gcs_resource():
    """Access GCS using the mounted Kubernetes secret."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

def get_model_history():
    """Fetch model versions from the tracking database."""
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        versions = client.search_model_versions("name='HousingPriceModel'")
        history = []
        for v in versions:
            try:
                run = client.get_run(v.run_id)
                acc = run.data.metrics.get("accuracy", "N/A")
            except:
                acc = "N/A"
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "accuracy": f"{acc:.4f}" if isinstance(acc, float) else acc,
                "created": datetime.fromtimestamp(v.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M')
            })
        return sorted(history, key=lambda x: int(x['version']), reverse=True)
    except Exception:
        return []

@app.route('/')
@app.route('/dashboard')
def dashboard():
    try:
        _, bucket = get_gcs_resource()
        blobs = bucket.list_blobs()
        files = []
        for blob in blobs:
            if any(blob.name.startswith(p) for p in ['data/', 'uploads/', 'models/']):
                if blob.name.endswith('/'): continue
                files.append({
                    'name': blob.name,
                    'size': f"{blob.size / 1024:.2f} KB",
                    'time': blob.updated.strftime('%Y-%m-%d %H:%M')
                })
        return render_template('dashboard.html', files=files, model_versions=get_model_history())
    except Exception as e:
        return f"Dashboard Error: {e}", 500

@app.route('/trigger-train', methods=['POST'])
def trigger_train():
    """Run train.py in background and pipe output to log file."""
    try:
        # 1. Clear/Create the log file
        with open(LOG_FILE_PATH, "w") as f:
            f.write(f"--- NEW TRAINING SESSION: {datetime.now()} ---\n")
            f.flush()

        # 2. Start the process
        log_f = open(LOG_FILE_PATH, "a")
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = TRACKING_URI
        
        subprocess.Popen(
            ["python3", "-u", "/app/train.py"], 
            stdout=log_f, 
            stderr=log_f,
            env=env
        )
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Training failed to start: {e}", 500

@app.route('/logs')
def get_logs():
    """Read logs for the frontend terminal."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            return f.read()
    return "Initializing logs..."

@app.route('/promote/<int:version>', methods=['POST'])
def promote_model(version):
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        client.transition_model_version_stage("HousingPriceModel", version, "Production", archive_existing_versions=True)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Promotion error: {e}", 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        _, bucket = get_gcs_resource()
        blob = bucket.blob(filename)
        file_data = io.BytesIO()
        blob.download_to_file(file_data)
        file_data.seek(0)
        return send_file(file_data, as_attachment=True, download_name=filename.split('/')[-1])
    except Exception as e:
        return f"Download failed: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)