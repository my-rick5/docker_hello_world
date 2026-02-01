import os
import threading
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify
from google.cloud import storage

app = Flask(__name__)

# --- GCS Configuration ---
BUCKET_NAME = os.environ.get('GCP_BUCKET_NAME', 'housing-data-for-testing')

def get_gcs_resource():
    """Helper to initialize the GCS client."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

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
        return render_template('dashboard.html', files=file_list)
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
            subprocess.run(["python", "train.py"], check=True)
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

if __name__ == "__main__":
    # Crucial for GKE: Listen on 8080
    app.run(host='0.0.0.0', port=8080)