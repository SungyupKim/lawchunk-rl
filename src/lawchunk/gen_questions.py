"""조문별 일상어 합성 질문 생성 (Claude API). ANTHROPIC_API_KEY 필요.

조문 문구를 베끼지 않고 법령 용어 없이 일반인이 묻는 말투로 쓰게 한다.
사용: python -m lawchunk.gen_questions 근로기준법 [조문당질문수=3]
"""
import json
import random
import sys
from pathlib import Path

from .chunk import article_text, load_articles

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-5-5"
PROMPT = """다음은 {law}의 한 조문입니다.

{text}

이 조문이 답이 되는, 일반인이 일상어로 묻는 질문 {n}개를 만드세요.
- 조문에 나온 법령 용어/표현("사용자", "근로자", "소정" 등)과 조 번호를 그대로 쓰지 말고, 실생활 상황으로 풀어 쓸 것
- 이 조문만으로 답할 수 있어야 하고, 질문마다 서로 다른 측면을 물을 것
- JSON 문자열 배열만 출력"""


def main(law: str, n: int = 3, seed: int = 0):
    import anthropic

    client = anthropic.Anthropic()
    arts = [a for a in load_articles(law) if not article_text(a).split("\n", 1)[-1].strip().startswith("삭제")]
    arts = [a for a in arts if len(article_text(a)) > 60]
    out = ROOT / "data" / "questions" / f"{law}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with out.open("w", encoding="utf-8") as f:
        for a in arts:
            msg = client.messages.create(
                model=MODEL, max_tokens=800,
                messages=[{"role": "user", "content": PROMPT.format(law=law, text=article_text(a), n=n)}],
            )
            raw = msg.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
            split = "test" if rng.random() < 0.3 else "dev"  # 조문 단위로 분할 (누수 방지)
            for q in json.loads(raw):
                f.write(json.dumps({"q": q, "gold": [a["id"]], "split": split}, ensure_ascii=False) + "\n")
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
