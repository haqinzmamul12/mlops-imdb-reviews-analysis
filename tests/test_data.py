import pandas as pd
from src.data.load_data import load_imdb_data


def test_load_imdb_data():
    df = load_imdb_data("raw")
    assert isinstance(df, pd.DataFrame), "Loaded data should be a DataFrame"
    assert "review" in df.columns, "DataFrame must contain 'review' column"
    assert "sentiment" in df.columns, \
    "DataFrame must contain 'sentiment' column"
    assert len(df) > 0, "DataFrame must not be empty"


test_load_imdb_data()
