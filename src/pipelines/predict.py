from src.utils.load_artifacts import load_model 
import yaml 

def predict_sentiment(reviews):
    try:
        with open("config/config.yaml") as f:
            config =yaml.safe_load(f)

        vectorizer_path = config['model']['vectorizer_path']
        model_path =config['model']['model_path']

        vectorizer =load_model(vectorizer_path)
        model =load_model(model_path)

        X =vectorizer.transform(reviews)
        preds =model.predict(X)
        labels = ["positive" if p == 1 else "negative" for p in preds]
        return labels
     
    except Exception as e:
        print(f"Error Encountered: {repr(e)}")


