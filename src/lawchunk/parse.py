"""법령 JSON -> 조/항/호 계층 구조 + 상호 참조 추출.

사용법: python -m lawchunk.parse 근로기준법
"""
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 「다른 법」 제N조 는 외부 참조, 그 외 "제N조(의M)(제K항)(제L호)" 는 내부 참조로 본다.
REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?(?:제(\d+)항)?(?:제(\d+)호)?")
EXT_RE = re.compile(r"「[^」]+」\s*$")


@dataclass
class Item:  # 호
    no: str
    text: str


@dataclass
class Paragraph:  # 항
    no: str
    text: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Article:  # 조
    id: str  # 예: 제10조의2
    title: str
    chapter: str
    heading: str  # 예: 제2조(정의)
    paragraphs: list[Paragraph]
    refs: list[str]  # 내부 참조 조문 id 목록

    def full_text(self) -> str:
        lines = [self.heading]
        for p in self.paragraphs:
            if p.text:
                lines.append(p.text)
            lines += [i.text for i in p.items]
        return "\n".join(lines)


def _as_list(x):
    if x is None or x == "":
        return []
    return x if isinstance(x, list) else [x]


def _flat(x) -> str:
    """호 아래 목 등 더 깊은 구조는 한 덩어리 문자열로 편다."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return "\n".join(_flat(i) for i in x)
    if isinstance(x, dict):
        return "\n".join(_flat(v) for k, v in x.items() if k.endswith("내용") or k in ("목", "목단위"))
    return ""


def _article_id(u: dict) -> str:
    n = u["조문번호"]
    b = u.get("조문가지번호")
    return f"제{n}조" + (f"의{b}" if b else "")


def _refs(text: str, self_id: str) -> list[str]:
    out: list[str] = []
    for m in REF_RE.finditer(text):
        # 바로 앞이 「법령명」 이면 외부 법령 참조이므로 제외
        if EXT_RE.search(text[: m.start()]):
            continue
        aid = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        if aid != self_id and aid not in out:
            out.append(aid)
    return out


def parse_law(path: Path) -> list[Article]:
    units = json.loads(path.read_text(encoding="utf-8"))["법령"]["조문"]["조문단위"]
    articles, chapter = [], ""
    for u in units:
        if u["조문여부"] == "전문":
            chapter = " ".join(u["조문내용"].split())
            continue
        paras = []
        for h in _as_list(u.get("항")):
            items = [Item(i.get("호번호", "").strip(), _flat(i.get("호내용", ""))) for i in _as_list(h.get("호"))]
            paras.append(Paragraph(h.get("항번호", "").strip(), _flat(h.get("항내용", "")), items))
        heading = " ".join(u["조문내용"].split()) if paras else ""
        aid = _article_id(u)
        title = u.get("조문제목", "")
        heading = f"{aid}({title})" if title else aid
        # 조문내용에서 '제N조(제목)' 접두어를 떼고 남는 글이 있으면 본문(리드문)이다.
        # 항이 없는 조문, 또는 항에 호만 있는 조문(예: 제26조)에서 나타난다.
        lead = " ".join(_flat(u["조문내용"]).split()).removeprefix(heading).strip()
        if lead:
            paras.insert(0, Paragraph("", lead))
        art = Article(aid, title, chapter, heading, paras, [])
        art.refs = _refs(art.full_text(), aid)
        articles.append(art)
    return articles


def main(name: str):
    arts = parse_law(ROOT / "data" / "raw" / f"{name}.json")
    out = ROOT / "data" / "processed" / f"{name}.articles.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a in arts:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    n_p = sum(len(a.paragraphs) for a in arts)
    n_i = sum(len(p.items) for a in arts for p in a.paragraphs)
    n_r = sum(len(a.refs) for a in arts)
    print(f"{name}: 조 {len(arts)}, 항 {n_p}, 호 {n_i}, 내부참조 {n_r} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "근로기준법")
