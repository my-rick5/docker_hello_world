import os
import io
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from google.cloud import storage

# --- Configuration ---
# Uses Environment Variable first (memory), falls back to Persistent Volume if needed
DB_PATH = "/app/data/mlflow.db"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DB_PATH}")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")

def get_gcs_resource():
    """Initialize GCS client using the mounted secret key."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

def train_model():
    # 1. Setup MLflow tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Housing_Price_Analysis")

    # The flush=True is critical for the dashboard to show logs in real-time
    print(f"Connecting to GCS bucket: {BUCKET_NAME}...", flush=True)
    try:
        client, bucket = get_gcs_resource()
    except Exception as e:
        print(f"❌ GCS Connection Error: {e}", flush=True)
        return

    # 2. Pull data from the 'data/' folder in GCS
    blob_name = "data/test_data.csv"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        print(f"❌ Error: {blob_name} not found in bucket. Please upload via Dashboard.", flush=True)
        return

    print(f"Downloading {blob_name}...", flush=True)
    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data_bytes))

    # 3. Preprocessing
    if 'sqft' in df.columns and 'price' in df.columns:
        X = df[['sqft']]
        y = df['price']
    else:
        print("Columns 'sqft'/'price' not found. Using fallback columns.", flush=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training with MLflow Tracking
    with mlflow.start_run(run_name="GKE_Background_Train"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        accuracy = 1.0 / (1.0 + mse) 
        
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(
            model, 
            "model", 
            registered_model_name="HousingPriceModel"
        )
        
        print(f"✅ Training complete. Accuracy: {accuracy:.4f}", flush=True)
        print("✅ Training complete and model registered!", flush=True)

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}", flush=True)
        sys.exit(1)