import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import TextPreprocessor

PREPROCESSOR_CONFIG = {
    'lowercase': True,
    'remove_urls': True,
    'remove_html': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatization': True
}

def load_amazon(path="data/raw/Reviews.csv", n=50000):
    df = pd.read_csv(path, usecols=["Text", "Score"])
    df = df.dropna()
    df["label"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    df = df[df["Score"] != 3]
    return df[["Text", "label"]].sample(n=n, random_state=42).reset_index(drop=True)

def train_and_save():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nlp_lab1_production")

    print("Loading Amazon dataset...")
    df = load_amazon()

    preprocessor = TextPreprocessor(PREPROCESSOR_CONFIG)
    print("Preprocessing...")
    df["processed"] = df["Text"].apply(lambda x: preprocessor.process(str(x)))

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"], df["label"], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    with mlflow.start_run(run_name="production_classifier"):
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("dataset", "amazon")
        mlflow.log_param("n_samples", len(df))

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "classifier")

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        # save locally for API
        os.makedirs("models", exist_ok=True)
        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print("Models saved to models/")
        return model, vectorizer

if __name__ == "__main__":
    train_and_save()