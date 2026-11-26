"""고정 임베딩 모델 + 텍스트 해시 기반 디스크 캐시. 실험 중 모델을 바꾸지 말 것."""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "BAAI/bge-m3"  # 대안: nlpai-lab/KURE-v1
_model = None


def _get():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": "float16"})
        _model.max_seq_length = 1024
    return _model


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """정규화된 임베딩 (N, D). 캐시에 없는 텍스트만 계산한다."""
    cache = ROOT / "data" / "cache" / MODEL.replace("/", "_")
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / (hashlib.sha1(t.encode()).hexdigest() + ".npy") for t in texts]
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if todo:
        vecs = _get().encode([texts[i] for i in todo], batch_size=batch_size, normalize_embeddings=True)
        for i, v in zip(todo, vecs):
            np.save(paths[i], v.astype(np.float32))
    return np.stack([np.load(p) for p in paths])
