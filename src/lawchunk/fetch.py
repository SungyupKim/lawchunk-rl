"""국가법령정보센터 Open API로 법령 본문(JSON)을 내려받는다.

사용법: python -m lawchunk.fetch 근로기준법
인증키는 환경변수 LAW_OC 또는 프로젝트 루트의 .env 에서 읽는다.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://www.law.go.kr/DRF"


def _oc() -> str:
    if os.environ.get("LAW_OC"):
        return os.environ["LAW_OC"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LAW_OC="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("LAW_OC 가 없습니다. .env 에 LAW_OC=... 를 넣으세요.")


def _get(endpoint: str, **params) -> dict:
    params = {"OC": _oc(), "type": "JSON", **params}
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def fetch_law(name: str, out_dir: Path = ROOT / "data" / "raw") -> Path:
    res = _get("lawSearch.do", target="law", query=name)["LawSearch"]["law"]
    if isinstance(res, dict):
        res = [res]
    hit = next((x for x in res if x["법령명한글"] == name and x["현행연혁코드"] == "현행"), None)
    if hit is None:
        raise SystemExit(f"'{name}' 와 정확히 일치하는 현행 법령이 없습니다: {[x['법령명한글'] for x in res]}")
    body = _get("lawService.do", target="law", MST=hit["법령일련번호"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: MST={hit['법령일련번호']} 시행일자={hit['시행일자']} -> {path}")
    return path


if __name__ == "__main__":
    for n in sys.argv[1:] or ["근로기준법"]:
        fetch_law(n)
