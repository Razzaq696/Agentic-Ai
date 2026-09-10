"""Local Semantic Embedding module for private knowledge base retrieval."""

import json
import math
import os
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional
from langchain_core.embeddings import Embeddings


class LocalSemanticEmbeddings(Embeddings):
    """A lightweight, deterministic local semantic embedding model.

    Generates dense L2-normalized vector representations combining stemmed word tokens,
    subword character n-grams, and bigrams weighted by Inverse Document Frequency (IDF).
    Requires zero external API keys and runs completely offline.
    """

    def __init__(self, dim: int = 1024, state_path: Optional[str | Path] = None):
        self.dim = dim
        self.state_path = Path(state_path) if state_path else None
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "what", "who", "how", "when",
            "where", "which", "can", "should", "does", "do", "all", "any", "this"
        }
        if self.state_path and self.state_path.exists():
            self.load_state(self.state_path)

    def _stem(self, word: str) -> str:
        """Rule-based suffix stemmer to unify word variants."""
        w = word.lower()
        suffixes = [
            "ments", "ment", "ations", "ation", "tions", "tion",
            "ships", "ship", "ing", "ies", "ied", "es", "ed", "s"
        ]
        for suffix in suffixes:
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercased stemmed words."""
        raw_tokens = [
            t for t in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(t) > 1 and t not in self.stop_words
        ]
        return [self._stem(t) for t in raw_tokens]

    def _extract_features(self, text: str) -> List[str]:
        """Extract multi-granularity textual features (words, n-grams, bigrams)."""
        words = self._tokenize(text)
        features: List[str] = []

        for w in words:
            features.append(f"w_{w}")
            # Subword character n-grams (3 to 5 characters)
            for n in range(3, min(6, len(w) + 1)):
                for i in range(len(w) - n + 1):
                    features.append(f"ng_{w[i:i+n]}")

        # Adjacent word bigrams for phrase semantics
        for i in range(len(words) - 1):
            features.append(f"bg_{words[i]}_{words[i+1]}")

        return features

    def fit(self, texts: List[str]):
        """Fit vocabulary document frequencies from a list of corpus texts."""
        self.total_docs = len(texts)
        self.doc_freq.clear()
        for text in texts:
            unique_feats = set(self._extract_features(text))
            for f in unique_feats:
                self.doc_freq[f] += 1
        if self.state_path:
            self.save_state(self.state_path)

    def save_state(self, path: Path):
        """Save document frequency statistics to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "dim": self.dim,
                    "total_docs": self.total_docs,
                    "doc_freq": dict(self.doc_freq),
                }, f)
        except Exception as e:
            print(f"[Warning] Failed to save embedding state: {e}")

    def load_state(self, path: Path):
        """Load document frequency statistics from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.dim = data.get("dim", self.dim)
                self.total_docs = data.get("total_docs", 0)
                self.doc_freq = defaultdict(int, data.get("doc_freq", {}))
        except Exception as e:
            print(f"[Warning] Failed to load embedding state: {e}")

    def _text_to_vector(self, text: str) -> List[float]:
        """Convert an input text string into a normalized dense embedding vector."""
        features = self._extract_features(text)
        if not features:
            return [0.0] * self.dim

        counts = Counter(features)
        vec = [0.0] * self.dim

        for feat, count in counts.items():
            # Hash to dimension index and sign bit
            h = int(hashlib.sha256(feat.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0

            # Sublinear Term Frequency + BM25-style IDF weighting
            tf = 1.0 + math.log(count)
            df = self.doc_freq.get(feat, 0)
            idf = math.log((self.total_docs + 1.0) / (df + 1.0)) + 1.0
            weight = tf * idf

            # Full words and bigrams receive priority weighting over character n-grams
            if feat.startswith("w_"):
                weight *= 3.0
            elif feat.startswith("bg_"):
                weight *= 2.0

            vec[idx] += sign * weight

        # L2 Vector Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not self.doc_freq:
            self.fit(texts)
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._text_to_vector(text)


def get_embedding_model(persist_directory: Optional[str | Path] = None) -> LocalSemanticEmbeddings:
    """Factory function to get a configured embedding instance."""
    if persist_directory:
        state_file = Path(persist_directory) / "embedding_vocab.json"
        return LocalSemanticEmbeddings(dim=1024, state_path=state_file)
    return LocalSemanticEmbeddings(dim=1024)


if __name__ == "__main__":
    emb = get_embedding_model()
    test_docs = [
        "Students must maintain 80% attendance in lectures.",
        "Merit scholarships are awarded to students with GPA 3.75."
    ]
    vecs = emb.embed_documents(test_docs)
    q_vec = emb.embed_query("What is the attendance policy?")
    print(f"Generated {len(vecs)} document vectors of dimension {len(vecs[0])}")
    print(f"Generated query vector of dimension {len(q_vec)}")
