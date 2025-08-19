import pandas as pd
from src.utils.load_artifacts import load_model 
from sklearn.linear_model import LogisticRegression
import yaml 

def test_model_training_and_prediction():
    # Sample cleaned data
    df = pd.DataFrame({
        "cleaned_review": ["good movie", "bad movie", "excellent plot", "terrible acting"],
        "sentiment": ["positive", "negative", "positive", "negative"]
    })
    # Build features
    with open("config/config.yaml") as f:
        config =yaml.safe_load(f)

    vectorizer_path = config['model']['vectorizer_path']
    vectorizer =load_model(vectorizer_path)
    X =vectorizer.transform(df['cleaned_review'])
    y = df["sentiment"].apply(lambda x: 1 if x=="positive" else 0)

    
    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    # Predict
    test_reviews = ["good plot", "terrible movie"]
    vectorized_test_reviews =vectorizer.transform(test_reviews)
    predictions = model.predict(vectorized_test_reviews)

    predictions =['positive' if x else 'negative' for x in predictions]
    
    assert len(predictions) == len(test_reviews), "Number of predictions must match input"
    assert all(p in ["positive", "negative"] for p in predictions), "Predictions must be valid labels"

test_model_training_and_prediction()
