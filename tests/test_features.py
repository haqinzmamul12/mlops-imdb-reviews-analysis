import pandas as pd
from src.utils.load_artifacts import load_model
import yaml


def test_build_tfidf_features():
    # Sample cleaned data
    df = pd.DataFrame({"cleaned_review": ["good movie", "bad movie", "excellent plot"]})

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    vectorizer_path = config["model"]["vectorizer_path"]
    vectorizer = load_model(vectorizer_path)
    X = vectorizer.transform(df["cleaned_review"])

    # Tests
    assert X.shape[0] == 3, "Number of rows should match input DataFrame"
    assert X.shape[1] > 0, "TF-IDF feature columns should be > 0"
    assert hasattr(vectorizer, "transform"), "Vectorizer should have transform method"


test_build_tfidf_features()
