import os
import io
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from google.cloud import storage

# --- CONFIGURATION ---
TRACKING_URI = "http://mlflow:5000" 
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")

def get_gcs_resource():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/var/secrets/google/key.json"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return client, bucket

def train_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Housing_Price_Analysis")

    print(f"Connecting to GCS bucket: {BUCKET_NAME}...", flush=True)
    try:
        client, bucket = get_gcs_resource()
        blob_name = "data/test_data.csv"
        blob = bucket.blob(blob_name)
        data_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data_bytes))
    except Exception as e:
        print(f"⚠️ GCS Error/No File: {e}. Using dummy data for local test.", flush=True)
        df = pd.DataFrame({
            'sqft': np.random.randint(1000, 5000, 100),
            'price': np.random.randint(200000, 800000, 100)
        })

    # Data Cleaning
    df.columns = [c.lower().strip() for c in df.columns]
    if 'id' in df.columns: df = df.drop(columns=['id'])
    
    X = df[['sqft']] if 'sqft' in df.columns else df.iloc[:, :-1]
    y = df['price'] if 'price' in df.columns else df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Docker_Internal_Train") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions) # Better metric for UI visibility
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # 1. Log the model artifact
        model_info = mlflow.sklearn.log_model(
           sk_model=model, 
           artifact_path="model"
        )
        
        # 2. Explicitly Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "HousingPriceModel")
        
        print(f"✅ Training complete. R2 Score: {r2:.4f}", flush=True)
        print(f"✅ Model registered as 'HousingPriceModel' Version {run.info.run_id[:4]}", flush=True)

if __name__ == "__main__":
    train_model()