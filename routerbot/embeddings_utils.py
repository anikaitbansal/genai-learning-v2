import math
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_embedding_model = None

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model

def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()

def cosine_similarity(vector_a: list[float], vec_b: list[float]) -> float:
    if not vector_a or not vec_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)