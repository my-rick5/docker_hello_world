import os
import io
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from google.cloud import storage
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
DB_PATH = "/app/data/mlflow.db"
LOG_FILE_PATH = "/app/data/training.log"
TRACKING_URI = f"sqlite:///{DB_PATH}"

def get_gcs_resource():
    """Access GCS using the mounted Kubernetes secret."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

def get_model_history():
    """Fetch model versions from the persistent MLflow database."""
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        versions = client.search_model_versions("name='HousingPriceModel'")
        
        history = []
        for v in versions:
            # Try to get accuracy from the original run
            run = client.get_run(v.run_id)
            acc = run.data.metrics.get("accuracy", "N/A")
            
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "accuracy": f"{acc:.4f}" if isinstance(acc, float) else acc,
                "created": datetime.fromtimestamp(v.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M')
            })
        return sorted(history, key=lambda x: int(x['version']), reverse=True)
    except Exception:
        return []

# --- ROUTES ---

@app.route('/')
@app.route('/dashboard')
def dashboard():
    try:
        _, bucket = get_gcs_resource()
        blobs = bucket.list_blobs()
        
        files = []
        for blob in blobs:
            # STRICT FILTER: Hide all MLflow system noise
            if blob.name.startswith('mlflow-artifacts/') or "mlruns" in blob.name:
                continue
            
            # Only show your curated data and exports
            if any(blob.name.startswith(p) for p in ['data/', 'uploads/', 'models/']):
                if blob.name.endswith('/'): continue # Skip folder names
                
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
    """Run train.py in background and pipe output to the PVC log file."""
    try:
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write(f"\n\n--- NEW SESSION: {datetime.now()} ---\n")
            # Using -u for unbuffered output so logs stream in real-time
            subprocess.Popen(["python3", "-u", "train.py"], stdout=log_file, stderr=log_file)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Training failed to start: {e}", 500

@app.route('/logs')
def get_logs():
    """Read logs from the PVC for the frontend terminal."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            return "".join(f.readlines()[-100:]) # Return last 100 lines
    return "No logs yet. Click 'Retrain' to start."

@app.route('/promote/<int:version>', methods=['POST'])
def promote_model(version):
    """Promote a specific version and archive old production models."""
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        client.transition_model_version_stage(
            name="HousingPriceModel",
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Promotion error: {e}", 500

@app.route('/predict')
def predict():
    """Load the current 'Production' model dynamically."""
    sqft = request.args.get('sqft', 0)
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        # URI format: models:/<name>/<stage>
        model = mlflow.sklearn.load_model("models:/HousingPriceModel/Production")
        price = model.predict([[float(sqft)]])[0]
        return f"<h1>Estimate: ${price:,.2f}</h1><a href='/'>Back</a>"
    except Exception as e:
        return f"Prediction Error (Is a model promoted to Production?): {e}", 500

@app.route('/download/<path:filename>')
def download_file(filename): # This function name MUST match the url_for in HTML
    try:
        _, bucket = get_gcs_resource()
        blob = bucket.blob(filename)
        
        # Download the file to a buffer
        file_data = io.BytesIO()
        blob.download_to_file(file_data)
        file_data.seek(0)
        
        return send_file(
            file_data,
            as_attachment=True,
            download_name=filename.split('/')[-1] # Gets the actual filename
        )
    except Exception as e:
        return f"Download failed: {e}", 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)