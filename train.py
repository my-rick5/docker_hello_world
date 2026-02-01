import os
import io
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import storage
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient

# --- 1. CONFIGURATION ---
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"

BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
ARTIFACT_URI = f"gs://{BUCKET_NAME}/mlflow-artifacts"
MODEL_NAME = "HousingPriceModel" # The name for the Registry

# Setup MLflow Tracking
# Note: In a shared K8s environment, you'd eventually point this to a central server
mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "House_Price_Prediction"
try:
    mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
except Exception:
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
    with mlflow.start_run() as run:
        print(f"🔍 Active Run ID: {run.info.run_id}", flush=True)
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

        # --- 5. LOGGING & REGISTRATION ---
        
        # Log to MLflow Registry
        print(f"📦 Registering model as '{MODEL_NAME}'...", flush=True)
        try:
            # log_model and register_model_name handles versioning automatically
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="house_model",
                registered_model_name=MODEL_NAME
            )
            print("✅ Model registered successfully in MLflow.", flush=True)
        except Exception as e:
            print(f"⚠️ Registration failed: {e}", flush=True)

        # Legacy Support: Save specific pickle for the Flask app to GCS
        # (Optional: In a full MLOps setup, the Flask app would pull from MLflow)
        print("📤 Uploading legacy model.pkl to GCS...", flush=True)
        model_blob = bucket.blob("models/model.pkl")
        model_bytes = pickle.dumps(model)
        
        try:
            model_blob.upload_from_string(model_bytes)
            print("✅ Success! Legacy model saved to GCS.", flush=True)
        except Exception as e:
            print(f"❌ Legacy upload failed: {e}", flush=True)

if __name__ == "__main__":
    train_model()