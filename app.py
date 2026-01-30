import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template, redirect
import os
import pandas as pd


app = Flask(__name__)
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'housing_data.csv')
        
        # Simple Deduplication Logic: 
        # If the file exists, we merge and drop duplicates
        if os.path.exists(filepath):
            new_data = pd.read_csv(file)
            old_data = pd.read_csv(filepath)
            # Assuming 'id' is your pkey
            combined = pd.concat([old_data, new_data]).drop_duplicates(subset=['id'], keep='last')
            combined.to_csv(filepath, index=False)
        else:
            file.save(filepath)
            
        return redirect('/')
    return "Invalid file type. Please upload a CSV.", 400

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
