import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os  # Add this import

def train_model():
    # Load prepared data
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)  # Add this line
    
    # Save model
    joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    train_model()