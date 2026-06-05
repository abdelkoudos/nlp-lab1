import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from rank_bm25 import BM25Okapi
from src.preprocessor import TextPreprocessor

mlflow.set_tracking_uri("mlruns")

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

def load_twitter(path="data/raw/training.1600000.processed.noemoticon.csv", n=50000):
    df = pd.read_csv(path, encoding="latin-1", header=None,
                     names=["label","id","date","query","user","text"])
    df["label"] = df["label"].apply(lambda x: 1 if x == 4 else 0)
    return df[["text", "label"]].sample(n=n, random_state=42).reset_index(drop=True)

def run_experiment(dataset_name, texts, labels, vectorizer_name, vectorizer):
    preprocessor = TextPreprocessor(PREPROCESSOR_CONFIG)
    print(f"Preprocessing {dataset_name}...")
    processed = [preprocessor.process(str(t)) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"{dataset_name}_{vectorizer_name}"):
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("vectorizer", vectorizer_name)
        mlflow.log_param("n_samples", len(texts))

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"{dataset_name} | {vectorizer_name} | accuracy: {acc:.4f}")
        return acc, model, vectorizer

if __name__ == "__main__":
    mlflow.set_experiment("nlp_lab1")

    print("Loading datasets...")
    amazon_df = load_amazon()
    twitter_df = load_twitter()

    vectorizers = {
        "BoW": CountVectorizer(max_features=10000),
        "TF-IDF": TfidfVectorizer(max_features=10000),
    }

    for vec_name, vec in vectorizers.items():
        run_experiment("amazon", amazon_df["Text"], amazon_df["label"], vec_name, vec)

    for vec_name, vec in vectorizers.items():
        run_experiment("twitter", twitter_df["text"], twitter_df["label"], vec_name, vec)

    print("\nAll experiments done. Run: mlflow ui")