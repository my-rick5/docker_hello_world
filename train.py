import os
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from google.cloud import storage

# --- Configuration ---
# Match the persistent path from your PVC
DB_PATH = "/app/data/mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH}"
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")

def get_gcs_resource():
    """Initialize GCS client using the mounted secret key."""
    # The path to the key is defined in your deployment.yaml volume mount
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

def train_model():
    # 1. Setup MLflow tracking to the Persistent Volume
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Housing_Price_Analysis")

    print(f"Connecting to GCS bucket: {BUCKET_NAME}...")
    client, bucket = get_gcs_resource()

    # 2. Pull data from the ISOLATED 'data/' folder
    blob_name = "data/test_data.csv"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        print(f"❌ Error: {blob_name} not found in bucket. Please upload via Dashboard.")
        return

    print(f"Downloading {blob_name}...")
    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data_bytes))

    # 3. Simple Preprocessing (Assumes 'sqft' and 'price' columns)
    # Adjust these column names based on your actual test_data.csv
    if 'sqft' in df.columns and 'price' in df.columns:
        X = df[['sqft']]
        y = df['price']
    else:
        # Fallback for generic CSV testing
        print("Columns 'sqft'/'price' not found. Using first and last columns as features/target.")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training with MLflow Tracking
    with mlflow.start_run(run_name="GKE_Background_Train"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        # We'll log "accuracy" as 1 - normalized error for your dashboard display
        mse = mean_squared_error(y_test, predictions)
        accuracy = 1.0 / (1