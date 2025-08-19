from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.features.build_feature import FeatureBuilder
from src.utils.save_artifacts import dump_model
from src.data.load_data import load_imdb_data
import yaml
from src.features.text_cleaning import DataTransformation


class ModelTrainer:
    def __init__(self):
        self.builder = FeatureBuilder()
        self.transfomer = DataTransformation()
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.output_path = config["model"]["model_path"]

    def train_model(self):
        try:
            df_cleaned = load_imdb_data("cleaned")
            X, y = (
                df_cleaned.drop(columns=["sentiment"], axis=1),
                df_cleaned["sentiment"],
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            print("model trained successfully!")
            dump_model(self.output_path, model)

        except Exception as e:
            print(f"Error Encountered: {repr(e)}")
