import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os  # Added for directory handling

def evaluate_model():
    # Load model and test data
    model = joblib.load("models/model.joblib")
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop("target", axis=1)
    y_test = test["target"]
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Create directory if missing
    os.makedirs("results", exist_ok=True)  # Added this line
    
    # Save metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()