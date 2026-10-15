"""RAINY offline text clustering candidate for Apex SN1.

No models, network access, wallet access or training labels are used at runtime.
This is a research candidate, not a claim of competitive mining revenue.
"""
import argparse
import re
import threading
import time
import warnings

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import hstack
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

MAX_TEXTS = 50_000
MAX_CHARACTERS = 12_000_000
MAX_TEXT_LENGTH = 6_000
FIT_LIMIT = 5_500
WORK_BUDGET = 65.0
_LOCK = threading.Lock()
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION = re.compile(r"(?<!\w)@\w+")
_SPACE = re.compile(r"\s+")


class ClusterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[StrictStr] = Field(min_length=1, max_length=MAX_TEXTS)


class ClusterResponse(BaseModel):
    cluster_ids: list[int]


def preprocess_text(text):
    text = text[:MAX_TEXT_LENGTH]
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _SPACE.sub(" ", text.casefold()).strip()


def _features(texts):
    word = TfidfVectorizer(
        strip_accents="unicode", lowercase=False, sublinear_tf=True,
        ngram_range=(1, 2), min_df=1 if len(texts) < 30 else 2,
        max_df=1.0, max_features=45_000, stop_words="english",
        token_pattern=r"(?u)\b\w[\w-]+\b", dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=20_000, sublinear_tf=True, dtype=np.float32,
    )
    parts = []
    for vectorizer, weight in ((word, 1.0), (char, 0.30)):
        try:
            parts.append(vectorizer.fit_transform(texts) * weight)
        except ValueError:
            # Empty vocabulary is expected for punctuation-only / stopword input.
            continue
    if not parts:
        return None
    return normalize(hstack(parts, format="csr"), copy=False)


def _quality(vectors, labels, sample):
    chosen = labels[sample]
    valid = chosen >= 0
    coverage = float(np.mean(valid))
    if coverage < 0.35:
        return -2.0
    ids, counts = np.unique(chosen[valid], return_counts=True)
    if len(ids) < 2 or len(ids) >= valid.sum():
        return -2.0
    score = silhouette_score(vectors[sample][valid], chosen[valid], metric="cosine")
    # Reject solutions that improve silhouette by discarding most texts or
    # separating tiny islands from one giant cluster.
    singleton_rate = np.sum(counts[counts == 1]) / max(1, valid.sum())
    dominant = float(counts.max() / valid.sum())
    substantial = np.sum(counts >= max(2, int(valid.sum() * 0.01)))
    if dominant > 0.85 or substantial < 2:
        return -2.0
    return float(score * coverage - 0.12 * (1 - coverage)
                 - 0.08 * singleton_rate - 0.30 * max(0, dominant - 0.50))


def _cluster_unique(texts, started):
    n = len(texts)
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    sparse = _features(texts)
    if sparse is None:
        return np.zeros(n, dtype=np.int32)
    dimensions = min(96, sparse.shape[0] - 1, sparse.shape[1] - 1)
    rng = np.random.default_rng(812)
    fit_ids = np.sort(rng.choice(n, min(n, FIT_LIMIT), replace=False))
    if dimensions >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reducer = TruncatedSVD(n_components=dimensions, n_iter=4, random_state=812)
            reducer.fit(sparse[fit_ids])
            vectors = normalize(reducer.transform(sparse)).astype(np.float32)
    else:
        vectors = normalize(sparse.toarray()).astype(np.float32)
    del sparse
    fit = vectors[fit_ids]
    m = len(fit)
    sample = np.sort(rng.choice(m, min(m, 384), replace=False))
    # A bounded hierarchical search replaces the starter's fixed sqrt(n/2) K.
    distances = pdist(fit, metric="cosine")
    np.nan_to_num(distances, copy=False, nan=1.0, posinf=1.0, neginf=0.0)
    np.clip(distances, 0.0, 2.0, out=distances)
    tree = linkage(distances, method="average")
    del distances
    best_score = -3.0
    best = np.zeros(m, dtype=np.int32)
    candidates = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    for k in candidates:
        if k >= m or (k > 2 and time.monotonic() - started > WORK_BUDGET):
            break
        labels = fcluster(tree, t=k, criterion="maxclust").astype(np.int32) - 1
        score = _quality(fit, labels, sample)
        if score > best_score:
            best_score, best = score, labels
    # Balanced spherical partitions are useful on noisy text where average
    # linkage otherwise isolates a tiny island from one giant cluster.
    for k in (4, 8, 16, 32, 64, 96):
        if k >= m or time.monotonic() - started > WORK_BUDGET:
            break
        model = MiniBatchKMeans(n_clusters=k, batch_size=512, n_init=3,
                               max_iter=60, random_state=812)
        labels = model.fit_predict(fit)
        score = _quality(fit, labels, sample)
        if score > best_score:
            best_score, best = score, labels.astype(np.int32)
    if m >= 40 and time.monotonic() - started < WORK_BUDGET:
        # A density candidate can retain genuine noise and irregular group sizes.
        density = HDBSCAN(
            min_cluster_size=max(5, min(30, m // 100)), min_samples=3,
            metric="euclidean", n_jobs=2, copy=True,
        ).fit_predict(fit)
        score = _quality(fit, density, sample)
        if score > best_score:
            best_score, best = score, density.astype(np.int32)
    if n == m:
        return best
    # Large requests are learned on a bounded sample and assigned in chunks.
    ids = np.unique(best[best >= 0])
    if not len(ids):
        return np.zeros(n, dtype=np.int32)
    centers = normalize(np.stack([fit[best == label].mean(axis=0) for label in ids]))
    result = np.empty(n, dtype=np.int32)
    for start in range(0, n, 1024):
        result[start:start + 1024] = ids[np.argmax(vectors[start:start + 1024] @ centers.T, axis=1)]
    result[fit_ids] = best
    return result


def cluster_texts(texts):
    if not texts:
        return []
    if len(texts) > MAX_TEXTS or sum(map(len, texts)) > MAX_CHARACTERS:
        raise ValueError("Input exceeds the documented text/character limit")
    started = time.monotonic()
    unique, positions, indices = [], {}, []
    for text in texts:
        cleaned = preprocess_text(text)
        # Whitespace, links and punctuation without lexical content are noise.
        if not any(c.isalnum() for c in cleaned):
            indices.append(-1)
            continue
        if cleaned not in positions:
            positions[cleaned] = len(unique)
            unique.append(cleaned)
        indices.append(positions[cleaned])
    if not unique:
        return [-1] * len(texts)
    with threadpool_limits(limits=2):
        labels = _cluster_unique(unique, started)
    return [int(labels[i]) if i >= 0 else -1 for i in indices]


def make_app():
    app = FastAPI(title="RAINY Apex Text Cluster", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster(request: ClusterRequest):
        if sum(map(len, request.texts)) > MAX_CHARACTERS:
            raise HTTPException(413, "Text character budget exceeded")
        if not _LOCK.acquire(blocking=False):
            raise HTTPException(503, "Clustering worker is busy")
        try:
            return ClusterResponse(cluster_ids=cluster_texts(request.texts))
        finally:
            _LOCK.release()

    return app


app = make_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
