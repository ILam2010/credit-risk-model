import pandas as pd

def test_data_not_empty():
    df = pd.read_csv("data/raw/training.csv")
    assert df.shape[0] > 0


def test_columns_exist():
    df = pd.read_csv("data/raw/training.csv")
    required_cols = ["TransactionId", "AccountId", "Amount"]
    
    for col in required_cols:
        assert col in df.columns