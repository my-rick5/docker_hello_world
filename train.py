import os
import io
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import storage
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURATION ---
# Silence Git warnings and set GCS credentials
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"

BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
ARTIFACT_URI = f"gs://{BUCKET_NAME}/mlflow-artifacts"

# Setup MLflow Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# FIX: Ensure the experiment exists with the correct GCS artifact location
experiment_name = "House_Price_Prediction"
try:
    # Attempt to create experiment with GCS destination
    mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
except Exception:
    # If it exists, MLflow will use the location defined when it was first created
    mlflow.set_experiment(experiment_name)

def train_model():
    print(f"Checking for bucket: {BUCKET_NAME}...", flush=True)
    
    # --- 2. GCS CLIENT SETUP ---
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}", flush=True)
        return
    
    # --- 3. PULL DATA FROM GCS ---
    data_blob = bucket.blob("uploads/test_data.csv")
    
    if not data_blob.exists():
        print(f"❌ Error: No training data found at gs://{BUCKET_NAME}/uploads/test_data.csv", flush=True)
        return

    print("📥 Downloading training data from GCS...", flush=True)
    data_bytes = data_blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data_bytes))
    
    # --- 4. MLFLOW RUN ---
    with mlflow.start_run():
        # Debugging: Print where MLflow thinks it is sending things
        print(f"🔍 Active Run ID: {mlflow.active_run().info.run_id}", flush=True)
        print(f"🔍 Artifact Destination: {mlflow.get_artifact_uri()}", flush=True)
        print("🚀 Starting MLflow training run...", flush=True)
        
        # Hyperparameters
        c_param = 0.01
        mlflow.log_param("C_value", c_param)
        mlflow.log_param("model_type", "LogisticRegression")

        # Train the Model
        model = LogisticRegression(C=c_param)
        model.fit(df[['sqft']], df['is_expensive'])

        # Log Metrics
        accuracy = model.score(df[['sqft']], df['is_expensive'])
        mlflow.log_metric("accuracy", accuracy)
        print(f"📊 Model Accuracy: {accuracy}", flush=True)

        # --- 5. CLOUD SAVING ---
        
        # Save to MLflow Artifact Store (GCS)
        print("📦 Logging model to MLflow/GCS...", flush=True)
        try:
            mlflow.sklearn.log_model(model, "house_model")
            print("✅ MLflow artifacts logged successfully.", flush=True)
        except Exception as e:
            print(f"⚠️ MLflow artifact logging failed: {e}", flush=True)

        # Save specific pickle for the Flask app to GCS
        print("📤 Uploading model.pkl to GCS...", flush=True)
        model_blob = bucket.blob("models/model.pkl")
        model_bytes = pickle.dumps(model)
        
        try:
            model_blob.upload_from_string(model_bytes)
            print("✅ Success! Model saved to GCS (models/model.pkl)", flush=True)
        except Exception as e:
            print(f"❌ Upload failed: {e}", flush=True)

if __name__ == "__main__":
    train_model()