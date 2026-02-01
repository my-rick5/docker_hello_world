import os
import io
import threading
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from google.cloud import storage
from mlflow.tracking import MlflowClient

app = Flask(__name__)


# --- INITIALIZE CLIENT GLOBALLY ---
# This ensures every route (like /download) can access it
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
        # Get all versions of this model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for v in versions:
            # Fetch the run to get the accuracy metric
            run = client.get_run(v.run_id)
            accuracy = run.data.metrics.get("accuracy", "N/A")
            
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "accuracy": round(accuracy, 4) if isinstance(accuracy, float) else accuracy,
                "created": v.creation_timestamp # You can format this date
            })
    except Exception as e:
        print(f"MLflow Fetch Error: {e}")
        
    return sorted(history, key=lambda x: int(x['version']), reverse=True)


# --- Routes ---

@app.route('/')
def index():
    """Main page with Upload form and Predict form."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        _, bucket = get_gcs_resource()
        # REMOVE prefix="uploads/" to see all files in your bucket
        blobs = bucket.list_blobs() 
        file_list = []
        for blob in blobs:
            file_list.append({
                'name': blob.name,
                'size': f"{blob.size / 1024:.2f} KB",
                'time': blob.updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        model_versions = get_model_history()
        return render_template('dashboard.html', files=files, model_versions=model_versions)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/upload', methods=['POST'])
def upload():
    """Upload a CSV to the GCS 'uploads/' directory."""
    file = request.files.get('file')
    if not file:
        return "No file provided", 400
    
    try:
        file = request.files.get('file')
        _, bucket = get_gcs_resource()
        # Just use the filename directly
        blob = bucket.blob(file.filename) 
        blob.upload_from_file(file)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Upload failed: {e}", 500

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
    """Delete a specific file from the GCS bucket."""
    try:
        _, bucket = get_gcs_resource()
        # Re-attach the prefix since we stripped it for the UI
        full_path = f"uploads/{filename}"
        blob = bucket.blob(full_path)
        
        if blob.exists():
            blob.delete()
            print(f"Deleted {full_path} from GCS.")
            return redirect(url_for('dashboard'))
        else:
            return f"File {filename} not found in GCS", 404
    except Exception as e:
        return f"Error deleting file: {e}", 500

@app.route('/trigger-train', methods=['POST'])
def trigger_train():
    """Runs train.py in a background thread to prevent UI timeout."""
    def run_training():
        try:
            print("Background training started...")
            # This executes your train.py logic
            subprocess.run(["python", "./train.py"])
            print("Background training completed successfully.")
        except Exception as e:
            print(f"Background training failed: {e}")

    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({"status": "success", "message": "Training started."}), 202

@app.route('/predict', methods=['GET'])
def predict():
    """Pulls the latest model and makes a prediction."""
    # Placeholder for your prediction logic
    sqft = request.args.get('sqft')
    # logic to load model.pkl from GCS and predict...
    return f"Prediction for {sqft} sqft: (Mock: EXPENSIVE)"

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        # Now storage_client is recognized!
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        
        if not blob.exists():
            return "File not found in Cloud Storage", 404

        file_data = io.BytesIO()
        blob.download_to_file(file_data)
        file_data.seek(0)
        
        return send_file(
            file_data, 
            as_attachment=True, 
            download_name=filename.split('/')[-1]
        )
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return f"Internal Error: {e}", 500

@app.route('/promote/<version>', method=['POST'])
def promote_model(version):
    try:
        client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
        model_name = "HousingPriceModel"
        
        # Transition the chosen version to 'Production'
        # archive_existing_versions=True ensures only one model is 'Live' at a time
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"🚀 Model {model_name} v{version} is now LIVE.")
        return jsonify({"status": "success", "message": f"v{version} promoted to Production"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Crucial for GKE: Listen on 8080
    app.run(host='0.0.0.0', port=8080)