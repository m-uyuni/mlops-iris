from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def prepare_data():
    # Load Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    
    # Split into train/test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save datasets
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    prepare_data()