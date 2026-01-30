import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from google.cloud import storage
import io

app = Flask(__name__)

# --- 1. CLOUD STORAGE HELPER ---
def upload_to_gcs(file_bytes, filename, content_type):
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{filename}")
    
    blob.upload_from_string(file_bytes, content_type=content_type)
    print(f"✅ Uploaded {filename} to GCS.")

# --- 2. DYNAMIC TRAINING LOGIC ---
def train_on_uploaded_data(df):
    mlflow.set_experiment("House_Price_Prediction")
    
    with mlflow.start_run():
        # Using the columns from your CSV (assuming 'sqft' and 'is_expensive' exist)
        X = df[['sqft']]
        y = df['is_expensive']
        
        c_value = 0.01
        mlflow.log_param("C_regularization", c_value)
        
        model = LogisticRegression(C=c_value)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally for the /predict route
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return accuracy

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return "Please upload a valid CSV file", 400

    try:
        # Read file into memory once
        file_bytes = file.read()
        
        # 1. Send to Google Cloud (Archive)
        upload_to_gcs(file_bytes, file.filename, file.content_type)
        
        # 2. Turn bytes into a Dataframe for training
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        # 3. Train the model on this specific file
        accuracy = train_on_uploaded_data(df)
        
        return {
            "message": f"Success! Model trained on {file.filename}",
            "accuracy": accuracy,
            "gcs_status": "Uploaded"
        }, 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error processing file: {e}", 500

@app.route('/predict', methods=['GET'])
def predict():
    sqft = request.args.get('sqft')
    if not sqft:
        return "Missing 'sqft' parameter", 400

    if not os.path.exists('model.pkl'):
        return "No model trained yet. Please upload a CSV first.", 400

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        prediction = model.predict([[float(sqft)]])
        result = "EXPENSIVE" if prediction[0] == 1 else "AFFORDABLE"
        
        return {"sqft": sqft, "prediction": result}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)