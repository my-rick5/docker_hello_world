import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import mlflow.sklearn
import os

# 1. Setup MLflow Experiment
# This organizes your runs under one name in the UI
mlflow.set_experiment("House_Price_Prediction")
mlflow.set_tracking_uri("sqlite:///mlflow.db") 

def train_model():
    filepath = 'data/housing_data.csv'
    
    if not os.path.exists(filepath):
        print("No training data found in /data. Please upload a CSV first.")
        return

    df = pd.read_csv(filepath)
    
    with mlflow.start_run():
        print("Starting training run...")
        
        # 2. Simple Dataset
        # 0 = Cheap, 1 = Expensive
        data = {
            'sqft': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
            'is_expensive': [0, 0, 0, 1, 1, 1, 1, 1]
        }
        df = pd.DataFrame(data)

        # 3. Hyperparameters
        # We log these so we can compare different settings later
        c_param = .01
        mlflow.log_param("C_value", c_param)
        mlflow.log_param("model_type", "LogisticRegression")

        # 4. Train the Model
        model = LogisticRegression(C=c_param)
        model.fit(df[['sqft']], df['is_expensive'])

        # 5. Log Metrics
        # Accuracy is 1.0 if it predicts our dummy data perfectly
        accuracy = model.score(df[['sqft']], df['is_expensive'])
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy}")

        # 6. Save Model to MLflow
        # This allows MLflow to track the version of the model itself
        mlflow.sklearn.log_model(model, "house_model")

        # 7. Export as Pickle (For your Flask App)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        print("Model saved to model.pkl and logged to MLflow.")

if __name__ == "__main__":
    train_model()
