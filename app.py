import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

app = Flask(__name__)

# --- 1. HELPER FUNCTION (Not a route) ---
def upload_to_gcs(file_bytes, filename, content_type):
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
    
    # Initialize the client (Uses GOOGLE_APPLICATION_CREDENTIALS env var)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{filename}")

    # Upload the bytes we already read from the request
    blob.upload_from_string(
        file_bytes,
        content_type=content_type
    )
    print(f"✅ Successfully uploaded {filename} to {BUCKET_NAME}")
    return blob.public_url

# --- 2. THE ACTUAL FLASK ROUTE ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Read the file data into memory once
        # This is important because reading the stream twice would result in empty data
        file_bytes = file.read()
        content_type = file.content_type
        
        # Call our helper function
        upload_to_gcs(file_bytes, file.filename, content_type)
        
        return f"Success! '{file.filename}' is now in Google Cloud Storage.", 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {e}", 500

# --- 3. MLFLOW TRAINING (Runs once when app starts) ---
mlflow.set_experiment("House_Price_Prediction")

with mlflow.start_run():
    data = {'sqft': [500, 1000, 1500, 2000, 2500, 3000],
            'is_expensive': [0, 0, 0, 1, 1, 1]}
    df = pd.DataFrame(data)

    c_value = 0.01
    mlflow.log_param("C_regularization", c_value)

    model = LogisticRegression(C=c_value)
    model.fit(df[['sqft']], df['is_expensive'])

    accuracy = model.score(df[['sqft']], df['is_expensive'])
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)