import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from preprocessor import TextPreprocessor

app = FastAPI(
    title="NLP Lab1 — Sentiment & Search API",
    description="Sentiment classifier and BM25 search engine",
    version="1.0.0"
)

PREPROCESSOR_CONFIG = {
    'lowercase': True,
    'remove_urls': True,
    'remove_html': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatization': True
}

preprocessor = TextPreprocessor(PREPROCESSOR_CONFIG)

print("Loading classifier...")
with open("models/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Loading search engine...")
with open("models/search_engine.pkl", "rb") as f:
    data = pickle.load(f)
    from rank_bm25 import BM25Okapi
    bm25 = data["bm25"]
    documents = data["documents"]
    labels = data["labels"]

print("All models loaded. API ready.")

class ClassifyRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    sentiment_filter: int = None

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

@app.post("/classify")
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    processed = preprocessor.process(request.text)
    vec = vectorizer.transform([processed])
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    return {
        "text": request.text,
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(float(max(proba)), 4)
    }

@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    tokenized_query = preprocessor.process(request.query).split()
    scores = bm25.get_scores(tokenized_query)
    
    results = []
    for i, score in enumerate(scores):
        if request.sentiment_filter is not None:
            if labels[i] != request.sentiment_filter:
                continue
        results.append({
            "text": documents[i][:200],
            "score": round(float(score), 4),
            "sentiment": "positive" if labels[i] == 1 else "negative"
        })
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "query": request.query,
        "total_results": len(results),
        "results": results[:request.top_k]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)