"""청킹 전략별 검색 성능 (조문 단위 Recall@k, MRR).

질문 파일(jsonl): {"q": 질문, "gold": ["제23조", ...], "split": "dev"|"test"}
청크 점수를 조문 단위로 max-집계해 조문 랭킹을 만들고 gold와 비교한다.

사용: python -m lawchunk.evaluate 근로기준법 data/questions/근로기준법.jsonl [split]
"""
import json
import sys
from pathlib import Path

import numpy as np

from .chunk import STRATEGIES, load_articles
from .embed import embed

KS = (1, 3, 5, 10)


def rank_articles(q_vecs, chunks, c_vecs) -> list[list[str]]:
    sims = q_vecs @ c_vecs.T
    ids = [c.article_id for c in chunks]
    out = []
    for row in sims:
        seen, ranking = set(), []
        for j in np.argsort(-row):
            if ids[j] not in seen:
                seen.add(ids[j])
                ranking.append(ids[j])
        out.append(ranking)
    return out


def metrics(rankings, golds):
    res = {f"R@{k}": 0.0 for k in KS}
    res["MRR"] = 0.0
    for r, g in zip(rankings, golds):
        g = set(g)
        for k in KS:
            res[f"R@{k}"] += len(g & set(r[:k])) / len(g)  # 정답 조문 중 top-k에 든 비율
        rr = next((1 / (i + 1) for i, a in enumerate(r) if a in g), 0.0)
        res["MRR"] += rr
    return {k: v / len(golds) for k, v in res.items()}


def run(law: str, qpath: Path, split: str | None = None):
    arts = load_articles(law)
    qs = [json.loads(l) for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    if split:
        qs = [q for q in qs if q.get("split") == split]
    q_vecs = embed([q["q"] for q in qs])
    golds = [q["gold"] for q in qs]
    print(f"{law}: 질문 {len(qs)}개" + (f" (split={split})" if split else ""))
    print(f"{'전략':<22}{'청크':>5}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'MRR':>8}")
    for key, fn in STRATEGIES.items():
        for prefix in (False, True):
            chunks = fn(law, arts, prefix=prefix)
            c_vecs = embed([c.embed_text for c in chunks])
            m = metrics(rank_articles(q_vecs, chunks, c_vecs), golds)
            name = f"{key}{'+prefix' if prefix else ''}"
            print(f"{name:<22}{len(chunks):>5}" + "".join(f"{m[f'R@{k}']:>8.3f}" for k in KS) + f"{m['MRR']:>8.3f}")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
