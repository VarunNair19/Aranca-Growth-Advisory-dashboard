import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_logger, settings

logger = get_logger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = settings.embedding_model) -> SentenceTransformer:
    """Loads and caches the SentenceTransformer encoder once per process."""
    if model_name not in _MODEL_CACHE:
        try:
            logger.info("Loading embedding model: %s", model_name)
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info("Successfully loaded embedding model: %s", model_name)
        except Exception:
            logger.exception("Failed loading embedding model: %s", model_name)
            raise
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str], model_name: str = settings.embedding_model) -> list[list[float]]:
    """Computes L2-normalized embedding vectors for a non-empty list of texts."""
    if not texts:
        return []
    try:
        model = get_embedding_model(model_name)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        logger.debug("Generated %d embeddings (dim=%d)", arr.shape[0], arr.shape[1])
        return arr.tolist()
    except Exception:
        logger.exception("Failed generating embeddings for %d texts", len(texts))
        raise


def embed_single(text: str, model_name: str = settings.embedding_model) -> list[float]:
    """Computes a single L2-normalized embedding vector."""
    vecs = embed_texts([text], model_name=model_name)
    return vecs[0] if vecs else []


def cosine_similarities(
    query_vec: list[float],
    candidate_vecs: list[np.ndarray],
) -> list[float]:
    """Computes cosine similarity scores between a normalized query vector and candidate vectors."""
    if not query_vec or not candidate_vecs:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return [0.0 for _ in candidate_vecs]
    q = q / q_norm

    scores: list[float] = []
    for c in candidate_vecs:
        c_arr = np.asarray(c, dtype=np.float32)
        c_norm = np.linalg.norm(c_arr)
        if c_norm == 0.0 or c_arr.shape != q.shape:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q, c_arr / c_norm)))
    return scores


def reciprocal_rank_fusion(
    fts_ranked_ids: list[int],
    vector_ranked_ids: list[int],
    rrf_k: int = 60,
    top_n: int = 12,
) -> list[int]:
    """Fuses lexical FTS5 rankings and semantic vector rankings using RRF."""
    scores: dict[int, float] = {}

    for rank, doc_id in enumerate(fts_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc_id in enumerate(vector_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_n]]