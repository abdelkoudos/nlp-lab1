# NLP Lab 1 — Report: Intelligence System From Raw Text to Production

**Course:** NLP  
**Weight:** 10% of course grade  
**Due:** Saturday, May 30  
**Student:** Abdelkoudos Elmiggabbar  
**GitHub:** https://github.com/abdelkoudos/nlp-lab1

---

## 1. Overview

This lab builds a fully operational sentiment classifier and search engine on two real-world noisy datasets, designed, tracked, and deployed as a production-grade ML system using MLOps practices.

**Datasets:**
- Amazon Fine Food Reviews — 568K long-form product reviews, clean and structured
- Sentiment140 — 1.6M tweets, short, noisy, informal

The contrast between these two datasets is intentional — the same pipeline, same algorithms, same code produces very different results, forcing real engineering decisions.

---

## 2. Learning Objectives — Status

| Objective | Status | Evidence |
|-----------|--------|----------|
| Build a reusable NLP pipeline | ✅ | src/preprocessor.py — configurable TextPreprocessor class |
| Compare vectorization methods | ✅ | src/vectorizer.py — BoW vs TF-IDF on both datasets, logged to MLflow |
| Understand distributional meaning | ✅ | src/embeddings.py — Word2Vec + PCA plots |
| Version data and experiments | ✅ | DVC for data, MLflow for experiments |
| Build a working search engine | ✅ | src/search_engine.py — BM25 over 100K documents with sentiment filter |
| Deploy a production API | ✅ | api/main.py — FastAPI with Docker |

---

## 3. Pipeline Architecture

    Raw Text
        ↓
    TextPreprocessor
    (lowercase, remove URLs, HTML, punctuation, stopwords, lemmatize)
        ↓
    Vectorizer (BoW / TF-IDF / BM25)
        ↓
    Classifier (Logistic Regression)
    → Sentiment: Positive / Negative
        ↓
    MLflow
    (tracks all experiments, saves best model)
        ↓
    FastAPI
    (serves classifier + BM25 search engine)
        ↓
    Docker
    (containerized, production-ready)

---

## 4. Preprocessing Pipeline

Built a configurable `TextPreprocessor` class in `src/preprocessor.py` that adapts to different text domains via a config dictionary — no code rewriting needed between datasets.

**Config used:**
- lowercase: True
- remove_urls: True
- remove_html: True
- remove_punctuation: True
- remove_stopwords: True
- lemmatization: True

**Key engineering decision:** Used lemmatization over stemming. Amazon reviews are long and rich — lemmatization preserves meaning ("studies" → "study"). Stemming is faster but loses quality ("studies" → "studi"). For Twitter, the difference is minimal since tweets are short and noisy anyway.

---

## 5. Vectorization Experiments

### Results

| Dataset | Vectorizer | Accuracy |
|---------|-----------|----------|
| Amazon | BoW | 92.3% |
| Amazon | TF-IDF | 91.5% |
| Twitter | BoW | 75.1% |
| Twitter | TF-IDF | 75.8% |

All runs tracked in MLflow under experiment `nlp_lab1`.

### Finding 1 — Amazon vs Twitter (92% vs 75%)

| Factor | Amazon Reviews | Twitter Tweets |
|--------|---------------|----------------|
| Length | 100+ words | 10-15 words |
| Language | Clean, formal | Noisy, slang, emojis |
| Signal | Rich vocabulary | Sparse vocabulary |
| Structure | Full sentences | Abbreviations, hashtags |

More words = more signal for the model. Tweets are too short and noisy — the model does not have enough context to work with. The 75% ceiling is the data quality, not the algorithm.

### Finding 2 — BoW beat TF-IDF on Amazon (92.3% vs 91.5%)

Amazon reviews are clean and specific. Raw word counts were already informative enough. TF-IDF's penalty on common words did not help — the signal was already strong in the raw counts.

### Finding 3 — TF-IDF beat BoW on Twitter (75.8% vs 75.1%)

Tweets are full of filler words. TF-IDF's ability to reward rare meaningful words gave a small but consistent advantage. The gap is small because both methods struggle equally with short noisy text.

---

## 6. Production Classifier

Built in `src/classifier.py` using TF-IDF + Logistic Regression with bigrams.

**Improvement over baseline:** Added `ngram_range=(1,2)` to capture phrases like "not good" and "very bad" as features.

**Results:**
- Accuracy: 91.8%
- Positive class: precision 92%, recall 98%
- Negative class: precision 87%, recall 56%

**Class imbalance observation:** The dataset is heavily skewed toward positive reviews (~85% positive). This causes the model to miss 44% of negative reviews. A production fix would be class weighting or oversampling — outside scope for this lab.

Model saved to `models/classifier.pkl` and `models/vectorizer.pkl` for API use.

---

## 7. Word Embeddings Analysis

Trained Word2Vec on both corpora using gensim (vector_size=100, window=5, epochs=10).

### Amazon Embeddings

![Amazon Embeddings](notebooks/amazon_embeddings.png)

Clear semantic clustering visible:
- Food terms cluster: coffee, tea, chocolate, drink, water
- Pet food terms cluster separately: cat, dog, treat, food
- Sentiment words cluster: good, great, best, better
- Purchase terms cluster: buy, price, store, brand

Word2Vec successfully learned domain meaning from clean long-form text.

### Twitter Embeddings

![Twitter Embeddings](notebooks/twitter_embeddings.png)

Poor clustering visible:
- Most words pile into one dense blob
- No clear semantic groups
- "amp" isolated — HTML entity &amp; not fully cleaned
- "na" isolated — slang not seen enough times to cluster

Word2Vec struggles on short noisy informal text.

### Similar Words Comparison

| Word | Amazon | Twitter |
|------|--------|---------|
| good | decent, great, fantastic | great, sunny, saturday, lovely |
| bad | terrible, horrible, weird | head, woke, b, saturday |
| love | — | awesome, ur, thanks, cool |
| hate | — | w, there, hard, many |

Amazon: "bad" correctly neighbors "terrible", "horrible" — meaningful.
Twitter: "bad" neighbors "head", "woke", "saturday" — meaningless.

### What Classical Vectors Cannot Capture

| Limitation | Example |
|------------|---------|
| Synonyms | BoW treats "good" and "great" as completely different features |
| Context | "bank" (river) vs "bank" (money) — identical BoW vector |
| Meaning distance | BoW has no concept that "terrible" is closer to "bad" than to "food" |
| Morphology | BoW treats "run", "running", "ran" as 3 different words |

Word2Vec learns that "taste" and "flavor" are similar from context. BoW cannot do this.

---

## 8. BM25 Search Engine

Built in `src/search_engine.py` over 100K Amazon reviews.

**Why BM25 over TF-IDF for search:**

| | TF-IDF | BM25 |
|---|--------|------|
| Purpose | Classification | Retrieval |
| Document length | Ignores it | Penalizes very long docs |
| Term frequency | Linear | Diminishing returns |
| Better for search | No | Yes |

**Features:**
- Search over 100K documents ranked by relevance score
- Sentiment filter — return only positive or negative results
- Pre-built index saved to models/search_engine.pkl

**Test results:**
- Query "great coffee taste" → returned relevant positive coffee reviews with scores 7.5, 7.3, 7.2
- Query "bad product" with negative filter → returned relevant negative reviews only

---

## 9. API Endpoints

Built with FastAPI, served via uvicorn, containerized with Docker.

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| /health | GET | — | service status |
| /classify | POST | text | sentiment + confidence |
| /search | POST | query + filter | ranked results |

**Example classify response:**
```
Input:  "This coffee is absolutely amazing!"
Output: { "sentiment": "positive", "confidence": 0.97 }
```

**Example search response:**
```
Input:  { "query": "great coffee", "sentiment_filter": 1, "top_k": 3 }
Output: top 3 positive reviews mentioning coffee, ranked by BM25 score
```

---

## 10. MLOps

| Tool | Usage |
|------|-------|
| DVC | Tracks large data files — Reviews.csv, training CSV, database.sqlite |
| MLflow | Logs all experiment runs — params, metrics, model artifacts |
| Git | Version controls all code |
| Docker | Containerizes the full API for reproducible deployment |

**MLflow experiments:**
- nlp_lab1 — baseline vectorization experiments (4 runs)
- nlp_lab1_production — final production classifier (1 run)

---

## 11. Project Structure

    nlp_lab1/
    ├── data/
    │   ├── raw/               # DVC-tracked — not in Git
    │   └── processed/
    ├── models/
    │   ├── classifier.pkl
    │   ├── vectorizer.pkl
    │   ├── search_engine.pkl
    │   ├── word2vec_amazon.model
    │   └── word2vec_twitter.model
    ├── notebooks/
    │   ├── amazon_embeddings.png
    │   └── twitter_embeddings.png
    ├── src/
    │   ├── preprocessor.py
    │   ├── vectorizer.py
    │   ├── embeddings.py
    │   ├── classifier.py
    │   └── search_engine.py
    ├── api/
    │   └── main.py
    ├── mlruns/
    ├── Dockerfile
    ├── requirements.txt
    └── README.md