import os
import io
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import storage
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURATION ---
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
# This ensures MLflow artifacts (models/plots) go to GCS
ARTIFACT_URI = f"gs://{BUCKET_NAME}/mlflow-artifacts"

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House_Price_Prediction")

def train_model():
    # --- 2. GCS CLIENT SETUP ---
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # --- 3. PULL DATA FROM GCS ---
    # Looking for the file your app.py uploads
    data_blob = bucket.blob("uploads/test_data.csv")
    
    if not data_blob.exists():
        print(f"No training data found in GCS: gs://{BUCKET_NAME}/uploads/test_data.csv")
        return

    print("Downloading training data from GCS...")
    data_bytes = data_blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data_bytes))
    
    # --- 4. MLFLOW RUN ---
    with mlflow.start_run(artifact_location=ARTIFACT_URI):
        print("Starting training run...")
        
        # Hyperparameters
        c_param = 0.01
        mlflow.log_param("C_value", c_param)
        mlflow.log_param("model_type", "LogisticRegression")

        # Train the Model
        # Using the columns from your uploaded CSV
        model = LogisticRegression(C=c_param)
        model.fit(df[['sqft']], df['is_expensive'])

        # Log Metrics
        accuracy = model.score(df[['sqft']], df['is_expensive'])
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy}")

        # --- 5. CLOUD SAVING (No local files) ---
        
        # Save to MLflow Artifact Store (GCS)
        mlflow.sklearn.log_model(model, "house_model")

        # Save specific pickle for the Flask app to GCS
        print("Uploading model.pkl to GCS...")
        model_blob = bucket.blob("models/model.pkl")
        model_bytes = pickle.dumps(model)
        model_blob.upload_from_string(model_bytes)
            
        print("Success! Model saved to GCS (models/model.pkl) and tracked in MLflow.")

if __name__ == "__main__":
    train_model()