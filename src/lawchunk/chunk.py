"""청킹 베이스라인 3종.

- fixed:     고정 길이 윈도우 (구조 무시)
- article:   조 단위
- hier:      조+항 계층형 (항 단위로 검색하되, 조 전체를 컨텍스트로 돌려줌)

모든 청크는 근거 조문 id(`article_id`)를 가지므로 평가 라벨(조문 ID)과 바로 비교된다.
`prefix=True`면 임베딩 대상 텍스트 앞에 "법령명 장 제N조(제목)" 맥락 접두어를 붙인다.

토큰 길이는 `count_tokens`로 주입한다. 기본값은 글자 수(한국어에서 토큰 수의 대략적 대용치).
실험에서는 임베딩 모델의 토크나이저로 바꿔 끼울 것.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Chunk:
    id: str
    article_id: str  # 평가 라벨
    embed_text: str  # 임베딩/검색 대상
    context_text: str  # 검색 후 LLM에 넘길 텍스트 (hier는 조 전체)


def load_articles(name: str) -> list[dict]:
    path = ROOT / "data" / "processed" / f"{name}.articles.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def article_text(a: dict) -> str:
    lines = [a["heading"]]
    for p in a["paragraphs"]:
        if p["text"]:
            lines.append(p["text"])
        lines += [i["text"] for i in p["items"]]
    return "\n".join(lines)


def _prefix(law: str, a: dict) -> str:
    return f"{law} {a['chapter']} {a['heading']}".strip()


def chunk_article(law: str, arts: list[dict], prefix: bool = False) -> list[Chunk]:
    out = []
    for a in arts:
        text = article_text(a)
        embed = f"{_prefix(law, a)}\n{text}" if prefix else text
        out.append(Chunk(a["id"], a["id"], embed, text))
    return out


def chunk_hier(law: str, arts: list[dict], prefix: bool = False) -> list[Chunk]:
    """항 단위 청크(호는 소속 항에 포함). 항이 없는 조문은 조 전체가 한 청크."""
    out = []
    for a in arts:
        full = article_text(a)
        paras = [p for p in a["paragraphs"] if p["text"] or p["items"]]
        if len(paras) <= 1:
            embed = f"{_prefix(law, a)}\n{full}" if prefix else full
            out.append(Chunk(a["id"], a["id"], embed, full))
            continue
        for k, p in enumerate(paras):
            body = "\n".join([p["text"]] + [i["text"] for i in p["items"]]).strip()
            head = _prefix(law, a) if prefix else a["heading"]
            out.append(Chunk(f"{a['id']}#{k}", a["id"], f"{head}\n{body}", full))
    return out


def chunk_fixed(
    law: str,
    arts: list[dict],
    size: int = 512,
    overlap: int = 64,
    count_tokens: Callable[[str], int] = len,
    prefix: bool = False,
) -> list[Chunk]:
    """법령 전체를 이어 붙인 뒤 고정 길이로 자른다 (조 경계 무시).

    한 청크가 걸친 모든 조문이 정답 후보다. 라벨은 청크 안에서 가장 많은 글자를 차지한
    조문 하나로 두고, 걸친 조문 전체는 `context_text`의 헤더가 아니라 id에 기록한다.
    count_tokens가 글자 수가 아니면 size는 토큰 수로 해석되므로, 글자 단위 슬라이싱 대신
    조 단위 문자열을 이어 붙여 만든 텍스트를 이진 탐색으로 자른다.
    """
    pieces: list[tuple[str, str]] = []  # (조문 id, 글자)
    for a in arts:
        for ch in article_text(a) + "\n":
            pieces.append((a["id"], ch))
    chars = "".join(c for _, c in pieces)

    def cut(start: int) -> int:
        lo, hi = start + 1, len(chars)
        while lo < hi:  # size 토큰을 넘지 않는 최대 끝 위치
            mid = (lo + hi + 1) // 2
            if count_tokens(chars[start:mid]) <= size:
                lo = mid
            else:
                hi = mid - 1
        return lo

    out, start, n = [], 0, 0
    while start < len(chars):
        end = cut(start)
        text = chars[start:end].strip()
        if text:
            ids = [i for i, _ in pieces[start:end]]
            main = max(set(ids), key=ids.count)
            embed = f"{law}\n{text}" if prefix else text
            out.append(Chunk(f"fixed#{n}:{main}", main, embed, text))
            n += 1
        if end >= len(chars):
            break
        # overlap만큼 되돌아가되 전진은 보장
        back = end
        while back > start + 1 and count_tokens(chars[back:end]) < overlap:
            back -= 1
        start = max(back, start + 1)
    return out


def chunk_article_refs(law: str, arts: list[dict], prefix: bool = False, max_refs: int = 3) -> list[Chunk]:
    """조 단위 + 내부 참조 조문 본문을 임베딩 텍스트에 덧붙인다.

    라벨은 원래 조문 하나이므로, 참조된 조문(B)을 찾는 질문은 B 자신의 청크가 맡고
    A를 거쳐 B 내용을 묻는 질문은 A 청크가 맡는다. 평가의 gold가 [A, B]인 질문에서
    A 청크 하나가 두 조문을 함께 끌어오는지가 관건이다.
    """
    by_id = {a["id"]: a for a in arts}
    out = []
    for a in arts:
        text = article_text(a)
        extra = [article_text(by_id[r]) for r in a["refs"][:max_refs] if r in by_id]
        embed = "\n".join([text] + extra)
        if prefix:
            embed = f"{_prefix(law, a)}\n{embed}"
        out.append(Chunk(a["id"], a["id"], embed, embed))
    return out


STRATEGIES = {
    "fixed": chunk_fixed,
    "article": chunk_article,
    "hier": chunk_hier,
    "article+refs": chunk_article_refs,
}


def main(name: str = "근로기준법"):
    arts = load_articles(name)
    for key, fn in STRATEGIES.items():
        for prefix in (False, True):
            cs = fn(name, arts, prefix=prefix)
            lens = [len(c.embed_text) for c in cs]
            print(f"{key:8s} prefix={prefix!s:5s} 청크 {len(cs):4d}  평균 {sum(lens)/len(lens):6.0f}자  최대 {max(lens)}자")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "근로기준법")
