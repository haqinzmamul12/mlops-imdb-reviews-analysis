import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from src.utils.save_artifacts import dump_model
from src.data.load_data import load_imdb_data


class FeatureBuilder:
    def __init__(self):
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.vectorizer_path = config["model"]["vectorizer_path"]
        self.cleaned = config["data"]["cleaned_path"]

    def build_tfidf_features(self):
        try:
            print("collecting interim dataset...")
            df_interim = load_imdb_data("interim")

            print("interim data procesing...")
            vectorizer = TfidfVectorizer(max_features=300)
            X = vectorizer.fit_transform(df_interim["review"])
            df_interim["sentiment"] = df_interim["sentiment"].apply(
                lambda x: 1 if x == "positive" else 0
            )

            feature_names = vectorizer.get_feature_names_out()
            X_df = pd.DataFrame(X.toarray(), columns=feature_names)
            df_cleaned = pd.concat(
                [df_interim["sentiment"].reset_index(drop=True), X_df], axis=1
            )
            print(
                f"TF-IDF features built: {X_df.shape[1]-1} features + sentiment for {X_df.shape[0]} samples"
            )
            df_cleaned.to_csv(self.cleaned, index=False)
            dump_model(self.vectorizer_path, vectorizer)

        except Exception as e:
            print(f"Error Encountered: {repr(e)}")



