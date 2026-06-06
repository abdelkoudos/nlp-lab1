import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from preprocessor import TextPreprocessor

PREPROCESSOR_CONFIG = {
    'lowercase': True,
    'remove_urls': True,
    'remove_html': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatization': True
}

class SearchEngine:
    def __init__(self):
        self.preprocessor = TextPreprocessor(PREPROCESSOR_CONFIG)
        self.bm25 = None
        self.documents = []
        self.labels = []

    def build_index(self, texts, labels):
        print("Building BM25 index...")
        self.documents = list(texts)
        self.labels = list(labels)
        tokenized = [self.preprocessor.process(str(t)).split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Index built on {len(self.documents)} documents")

    def search(self, query, top_k=10, sentiment_filter=None):
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")

        tokenized_query = self.preprocessor.process(query).split()
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if sentiment_filter is not None:
                if self.labels[i] != sentiment_filter:
                    continue
            results.append({
                "text": self.documents[i][:200],
                "score": float(score),
                "sentiment": int(self.labels[i])
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path="models/search_engine.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25,
                        "documents": self.documents,
                        "labels": self.labels}, f)
        print(f"Search engine saved to {path}")

    def load(self, path="models/search_engine.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.labels = data["labels"]
        print(f"Search engine loaded — {len(self.documents)} documents")

if __name__ == "__main__":
    print("Loading Amazon dataset...")
    df = pd.read_csv("data/raw/Reviews.csv", usecols=["Text", "Score"])
    df = df.dropna()
    df["label"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    df = df[df["Score"] != 3]
    df = df.sample(n=100000, random_state=42).reset_index(drop=True)

    engine = SearchEngine()
    engine.build_index(df["Text"], df["label"])
    engine.save()

    print("\nTest search: 'great coffee taste'")
    results = engine.search("great coffee taste", top_k=3)
    for r in results:
        print(f"Score: {r['score']:.3f} | Sentiment: {'positive' if r['sentiment']==1 else 'negative'}")
        print(f"Text: {r['text'][:100]}\n")

    print("Test search with sentiment filter (negative only):")
    results = engine.search("bad product", top_k=3, sentiment_filter=0)
    for r in results:
        print(f"Score: {r['score']:.3f} | Sentiment: negative")
        print(f"Text: {r['text'][:100]}\n")