import joblib

def test_model_accuracy():
    with open("results/metrics.txt", "r") as f:
        accuracy = float(f.read().split(" ")[1])
    assert accuracy >= 0.9, "Model accuracy too low!"