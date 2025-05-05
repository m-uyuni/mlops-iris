import pandas as pd

def test_data_integrity():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    # Check for NaN values
    assert not train.isnull().any().any()
    assert not test.isnull().any().any()
    
    # Check target distribution
    assert len(train["target"].unique()) == 3  # Iris has 3 classes