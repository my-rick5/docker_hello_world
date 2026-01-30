import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template, redirect
import os
import pandas as pd
from google.cloud import storage


app = Flask(__name__)
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_to_gcs(file_stream, filename):
    # GCP Configuration - usually set via Env Vars
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "your-unique-bucket-name")
    
    # Initialize the client
    # Note: When running in GCP (Cloud Run/GKE), credentials are handled automatically
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{filename}")

    # Stream the file directly from the Flask request to GCS
    blob.upload_from_string(
        file_stream.read(),
        content_type=file_stream.content_type
    )
    return blob.public_url

# Inside your Flask route:
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    # NEW: Upload to Cloud instead of local 'data/' folder
    try:
        gcs_url = upload_to_gcs(file, file.filename)
        return f"File uploaded successfully to {gcs_url}", 200
    except Exception as e:
        return f"GCP Upload Failed: {str(e)}", 500

# 1. Start the MLflow experiment
mlflow.set_experiment("House_Price_Prediction")

with mlflow.start_run():
    # Dummy data
    data = {'sqft': [500, 1000, 1500, 2000, 2500, 3000],
            'is_expensive': [0, 0, 0, 1, 1, 1]}
    df = pd.DataFrame(data)

    # Hyperparameters (the "settings" you chose)
    c_value = .01
    mlflow.log_param("C_regularization", c_value)

    # 2. Train model
    model = LogisticRegression(C=c_value)
    model.fit(df[['sqft']], df['is_expensive'])

    # 3. Log the "Metric" (how good is it?)
    accuracy = model.score(df[['sqft']], df['is_expensive'])
    mlflow.log_metric("accuracy", accuracy)

    # 4. Save the model to MLflow (The "Universal Adapter")
    mlflow.sklearn.log_model(model, "model")

    # Also save as pickle for your current Flask app
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
