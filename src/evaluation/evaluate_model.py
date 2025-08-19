import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.pipelines.train import ModelTrainer
from src.data.load_data import load_imdb_data
from sklearn.model_selection import train_test_split
import time


class Evaluater:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.MODEL_CANDIDATES = {
            "LogisticRegression": (
                LogisticRegression(max_iter=200),
                {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["liblinear", "saga"],
                },
            ),
            # "RandomForest": (
            # RandomForestClassifier(),
            # {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]}
            # ),
            "NaiveBayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}),
            "SVM": (LinearSVC(), {"C": [0.01, 0.1, 1, 10]}),
        }

    def evaluate_and_experiment(self):
        try:
            df_cleaned = load_imdb_data("cleaned")
            X, y = (
                df_cleaned.drop(columns=["sentiment"], axis=1),
                df_cleaned["sentiment"],
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            best_model = None
            best_score = 0
            best_model_name = None

            mlflow.set_experiment("IMDB-Sentiment-Analysis")

            for model_name, (model, model_dist) in self.MODEL_CANDIDATES.items():
                with mlflow.start_run(run_name=model_name) as run:
                    search = RandomizedSearchCV(
                        model,
                        param_distributions=model_dist,
                        scoring="accuracy",
                        cv=3,
                        n_jobs=-1,
                        random_state=42,
                    )

                    search.fit(X_train, y_train)
                    best_estimator = search.best_estimator_
                    y_pred = best_estimator.predict(X_test)

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Log to MLFlow
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metrics(
                        {
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1-score": f1,
                        }
                    )

                    # Log model into Registry
                    model_uri = f"runs:/{run.info.run_id}/{model_name}_model"
                    mlflow.sklearn.log_model(best_estimator, f"{model_name}_model")
                    mv = mlflow.register_model(model_uri, "IMDBClassifier")

                    print(f"{model_name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
                    print(f"Logged to MLflow Registry as version {mv.version}")

            # =========================
            # Polling step for approval
            # =========================
            print("⏳ Waiting for manual approval in MLflow UI...")
            approved_model = None
            while not approved_model:
                client = mlflow.tracking.MlflowClient()
                versions = client.search_model_versions("name='IMDBClassifier'")
                for v in versions:
                    # OLD: if v.current_stage == "Production":
                    if "challenger" in getattr(v, "aliases", []):  # check alias
                        approved_model = v
                        break
                if not approved_model:
                    time.sleep(15)  # poll every 15 sec

            print(
                f"✅ Approved model: Version {approved_model.version}, Stage: {approved_model.current_stage}"
            )

        except Exception as e:
            print(f"Error Encountered: {repr(e)}")
