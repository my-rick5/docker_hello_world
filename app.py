import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template, redirect, url_for
from sklearn.linear_model import LogisticRegression
from google.cloud import storage
import io
import re
from datetime import datetime

app = Flask(__name__)

# --- 1. DATA CLEANING & VALIDATION ---
def clean_and_validate(df, filename):
    df.columns = [col.strip().lower() for col in df.columns]
    required_cols = ['sqft', 'is_expensive']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in {filename}")

    for index, row in df.iterrows():
        for col in required_cols:
            val = str(row[col])
            bad_chars = re.findall(r'[^0-9.]', val)
            if bad_chars:
                line_num = index + 2
                raise ValueError(
                    f"Invalid character(s) '{''.join(set(bad_chars))}' found at line {line_num} "
                    f"in column '{col}' of {filename}. Value was: '{val}'"
                )
    
    df[required_cols] = df[required_cols].apply(pd.to_numeric)
    return df

# --- 2. CLOUD STORAGE HELPERS ---
def get_gcs_client():
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
    client = storage.Client()
    return client, client.bucket(BUCKET_NAME)

def upload_to_gcs(file_bytes, filename, content_type):
    _, bucket = get_gcs_client()
    blob = bucket.blob(f"uploads/{filename}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    print(f"✅ Uploaded {filename} to GCS.")

# --- 3. MEMORY TRAINING LOGIC ---
def train_with_memory(new_df, new_filename):
    mlflow.set_experiment("House_Price_Prediction")
    _, bucket = get_gcs_client()
    
    valid_new_df = clean_and_validate(new_df, new_filename)
    all_dataframes = [valid_new_df]
    
    blobs = bucket.list_blobs(prefix="uploads/")
    for blob in blobs:
        if blob.name.endswith(".csv") and not blob.name.endswith(new_filename):
            try:
                data = blob.download_as_bytes()
                old_df = pd.read_csv(io.BytesIO(data))
                all_dataframes.append(clean_and_validate(old_df, blob.name))
            except Exception as e:
                print(f"⚠️ Skipping legacy file {blob.name} due to error: {e}")
    
    master_df = pd.concat(all_dataframes).drop_duplicates()
    print(f"🧠 Model is training on {len(master_df)} total rows of memory.")

    with mlflow.start_run():
        X = master_df[['sqft']]
        y = master_df['is_expensive']
        model = LogisticRegression(C=0.01)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        mlflow.log_metric("total_rows", len(master_df))
        mlflow.log_metric("accuracy", accuracy)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return accuracy, len(master_df)

# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"status": "error", "message": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"status": "error", "message": "No selected file"}, 400

    try:
        file_bytes = file.read()
        new_df = pd.read_csv(io.BytesIO(file_bytes))
        accuracy, total_rows = train_with_memory(new_df, file.filename)
        upload_to_gcs(file_bytes, file.filename, file.content_type)
        
        return {
            "status": "success",
            "message": f"Model updated! It now remembers {total_rows} rows of data.",
            "accuracy": accuracy
        }, 200
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {e}"}, 500

@app.route('/dashboard')
def dashboard():
    try:
        BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "housing-data-for-testing")
        _, bucket = get_gcs_client()
        blobs = bucket.list_blobs(prefix="uploads/")
        
        file_list = []
        for blob in blobs:
            if blob.name.endswith(".csv"):
                file_list.append({
                    "name": blob.name.replace("uploads/", ""),
                    "full_path": blob.name,
                    "size": f"{round(blob.size / 1024, 2)} KB",
                    "updated": blob.updated.strftime('%Y-%m-%d %H:%M:%S')
                })
        file_list.sort(key=lambda x: x['updated'], reverse=True)
        return render_template('dashboard.html', files=file_list, bucket_name=BUCKET_NAME)
    except Exception as e:
        return f"Error loading dashboard: {e}", 500

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
    try:
        _, bucket = get_gcs_client()
        blob = bucket.blob(f"uploads/{filename}")
        blob.delete()
        print(f"🗑️ Deleted {filename} from GCS.")
        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Error deleting file: {e}", 500

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