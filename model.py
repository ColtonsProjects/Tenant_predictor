import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = 'tenant_model.joblib'
SCALER_PATH = 'tenant_scaler.joblib'

def train_model(data):
    # Convert list of lists to DataFrame
    df = pd.DataFrame(data, columns=[
        'MonthlyIncome',
        'FICOScore',
        'RentToIncomeRatio',
        'HasCriminalRecord', 
        'HasEvictionHistory',
        'AssetMonthlyValue',
        'ApplicationResult'
    ])
    
    # Separate features and target
    X = df[[
        'MonthlyIncome',
        'FICOScore',
        'RentToIncomeRatio',
        'HasCriminalRecord',
        'HasEvictionHistory', 
        'AssetMonthlyValue'
    ]]
    y = df['ApplicationResult']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Return accuracy
    accuracy = model.score(X_test_scaled, y_test)
    return float(accuracy)

def predict_tenant(tenant_data):
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Please train the model first.")
    
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare input data
    X = np.array(tenant_data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = int(model.predict(X_scaled)[0])
    confidence = float(max(model.predict_proba(X_scaled)[0]))
    
    return {
        'prediction': prediction,
        'confidence': confidence
    }

if __name__ == "__main__":
    print("Model module loaded successfully")
