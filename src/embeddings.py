import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from preprocessor import TextPreprocessor

PREPROCESSOR_CONFIG = {
    'lowercase': True,
    'remove_urls': True,
    'remove_html': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatization': True
}

def load_and_preprocess(path, text_col, n=10000):
    preprocessor = TextPreprocessor(PREPROCESSOR_CONFIG)
    if "Reviews" in path:
        df = pd.read_csv(path, usecols=[text_col, "Score"])
        df = df.dropna().sample(n=n, random_state=42)
    else:
        df = pd.read_csv(path, encoding="latin-1", header=None,
                        names=["label","id","date","query","user","text"])
        df = df.sample(n=n, random_state=42)
    
    print(f"Preprocessing {path}...")
    sentences = [preprocessor.process(str(t)).split() for t in df[text_col]]
    return [s for s in sentences if len(s) > 0]

def train_word2vec(sentences, name):
    print(f"Training Word2Vec on {name}...")
    model = Word2Vec(sentences, vector_size=100, window=5, 
                     min_count=2, workers=4, epochs=10)
    return model

def plot_embeddings(model, name, n_words=50):
    words = list(model.wv.key_to_index.keys())[:n_words]
    vectors = np.array([model.wv[w] for w in words])
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8)
    plt.title(f"Word2Vec embeddings — {name} (PCA)")
    plt.tight_layout()
    plt.savefig(f"notebooks/{name}_embeddings.png", dpi=100)
    print(f"Saved: notebooks/{name}_embeddings.png")
    plt.close()

def show_similar_words(model, words, name):
    print(f"\n--- Similar words ({name}) ---")
    for word in words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"{word}: {[w for w,_ in similar]}")

if __name__ == "__main__":
    # Amazon
    amazon_sentences = load_and_preprocess(
        "data/raw/Reviews.csv", "Text")
    amazon_model = train_word2vec(amazon_sentences, "amazon")
    plot_embeddings(amazon_model, "amazon")
    show_similar_words(amazon_model, ["good", "bad", "food", "taste"], "amazon")
    amazon_model.save("models/word2vec_amazon.model")

    # Twitter
    twitter_sentences = load_and_preprocess(
        "data/raw/training.1600000.processed.noemoticon.csv", "text")
    twitter_model = train_word2vec(twitter_sentences, "twitter")
    plot_embeddings(twitter_model, "twitter")
    show_similar_words(twitter_model, ["good", "bad", "love", "hate"], "twitter")
    twitter_model.save("models/word2vec_twitter.model")

    print("\nDone. Check notebooks/ for plots.")