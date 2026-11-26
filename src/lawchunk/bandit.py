"""조문별 '참조 조문을 붙일지(on/off)' 정책 학습 (REINFORCE).

- 행동: 참조가 있는 조문마다 a_i ∈ {0,1}. 1이면 article+refs 청크, 0이면 article 청크.
- 에피소드: 전 조문의 행동을 한 번 샘플링 → 학습용 질문의 MRR을 보상으로 받는다.
  (청크끼리 검색에서 경쟁하므로 보상은 코퍼스 전체 설정에 대한 값이다.)
- 정책: 조문 특징(길이, 참조 길이 등)의 로지스틱 회귀. 조문별 자유 파라미터가 없으므로
  학습에 쓰지 않은 조문(테스트)에도 그대로 적용된다.
- 학습/평가는 조문 단위로 분리: 테스트 조문을 정답으로 갖는 질문은 학습에서 제외한다.

사용: python -m lawchunk.bandit 민법
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .chunk import chunk_article, chunk_article_refs, load_articles
from .embed import embed

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA = 0.01  # 참조를 붙인 조문 비율에 대한 페널티 (청크 길이 증가 비용의 대용치)


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class Env:
    def __init__(self, law: str):
        arts = load_articles(law)
        self.ids = [a["id"] for a in arts]
        idx = {i: k for k, i in enumerate(self.ids)}
        plain = chunk_article(law, arts)
        refs = chunk_article_refs(law, arts)
        self.P = torch.tensor(embed([c.embed_text for c in plain]), device=DEV)
        self.R = torch.tensor(embed([c.embed_text for c in refs]), device=DEV)
        # 실제로 텍스트가 달라지는 조문만 행동 대상
        self.act = [k for k, (p, r) in enumerate(zip(plain, refs)) if p.embed_text != r.embed_text]
        self.feat = self._features(plain, refs)
        self.feat2 = self._features2(arts, plain, refs)

        qdir = ROOT / "data" / "questions"
        singles = _read(qdir / f"{law}.jsonl")
        multis = _read(qdir / f"{law}.multi.jsonl")
        test_art = {q["gold"][0] for q in singles if q["split"] == "test"}
        test_art |= {q["gold"][0] for q in multis if int(hashlib.sha1(q["gold"][0].encode()).hexdigest(), 16) % 10 < 3}
        self.test_art = test_art
        qs = [dict(q, kind="single") for q in singles] + [dict(q, kind="multi") for q in multis]
        tr = [q for q in qs if not set(q["gold"]) & test_art]
        te = [q for q in qs if set(q["gold"]) & test_art]
        self.train, self.test = self._pack(tr, idx), self._pack(te, idx)
        print(f"{law}: 행동 대상 조문 {len(self.act)}, 학습 질문 {len(tr)}, 테스트 질문 {len(te)}")

    def _features(self, plain, refs):
        f = []
        for k in self.act:
            own, full = len(plain[k].embed_text), len(refs[k].embed_text)
            f.append([np.log(own), np.log(full - own), np.log(full / own), (full - own) / 1000])
        f = np.array(f, dtype=np.float32)
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        return torch.tensor(np.hstack([np.ones((len(f), 1), np.float32), f]), device=DEV)

    def _features2(self, arts, plain, refs):
        """보강 특징: 기본 4개 + 청크 변화량(코사인), 참조 수, 준용/위임 문구, 참조 표기 비중, 본문 길이."""
        import re

        cue = re.compile(r"준용|에 따른|에 의한|에 의하여|의 규정|에 정한|에서 정한")
        refpat = re.compile(r"제\d+조(?:의\d+)?(?:제\d+항)?(?:제\d+호)?")
        f = []
        for k in self.act:
            a = arts[k]
            body = "\n".join(p["text"] for p in a["paragraphs"])
            own = len(plain[k].embed_text)
            ref_chars = sum(len(m.group(0)) for m in refpat.finditer(body))
            f.append([
                float((self.P[k] * self.R[k]).sum()),
                min(len(a["refs"]), 3),
                1.0 if cue.search(body) else 0.0,
                ref_chars / max(len(body), 1),
                np.log(max(len(body), 1)),
            ])
        f = np.array(f, dtype=np.float32)
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        return torch.cat([self.feat, torch.tensor(f, device=DEV)], dim=1)

    def _pack(self, qs, idx):
        Q = torch.tensor(embed([q["q"] for q in qs]), device=DEV)
        G = torch.zeros(len(qs), len(self.ids), dtype=torch.bool, device=DEV)
        for r, q in enumerate(qs):
            for g in q["gold"]:
                G[r, idx[g]] = True
        multi = torch.tensor([q["kind"] == "multi" for q in qs], device=DEV)
        return Q, G, multi

    def evaluate(self, a: torch.Tensor, split) -> dict:
        """a: 행동 대상 조문별 0/1 (len(self.act))."""
        Q, G, multi = split
        mask = torch.zeros(len(self.ids), dtype=torch.bool, device=DEV)
        mask[torch.tensor(self.act, device=DEV)[a.bool()]] = True
        C = torch.where(mask[:, None], self.R, self.P)
        sims = Q @ C.T
        best = sims.masked_fill(~G, -9).max(1).values
        rank = 1 + (sims > best[:, None]).sum(1)
        mrr = (1.0 / rank).float()
        top = sims.topk(10, dim=1).indices
        rec = G.gather(1, top).sum(1).float() / G.sum(1)
        return {"mrr": mrr.mean().item(), "r10": rec.mean().item(),
                "mrr_multi": mrr[multi].mean().item(), "mrr_single": mrr[~multi].mean().item()}


def train(env: Env, steps=300, batch=16, lr=0.1, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    theta = torch.zeros(env.feat.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for s in range(steps):
        p = torch.sigmoid(env.feat @ theta)
        acts = (torch.rand(batch, len(p), device=DEV, generator=g) < p).float()
        r = torch.tensor([env.evaluate(a, env.train)["mrr"] - LAMBDA * a.mean().item() for a in acts], device=DEV)
        adv = (r - r.mean()) / (r.std() + 1e-8)
        logp = (acts * torch.log(p + 1e-8) + (1 - acts) * torch.log(1 - p + 1e-8)).sum(1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return theta.detach()


def oracle(env: Env, split, passes=2):
    """주어진 질문 집합에 대한 좌표 상승(탐욕) 상한. 그 질문 위에서 튜닝하므로 낙관적이다."""
    a = torch.zeros(len(env.act), device=DEV)
    cur = env.evaluate(a, split)["mrr"]
    for _ in range(passes):
        for i in range(len(a)):
            a[i] = 1 - a[i]
            new = env.evaluate(a, split)["mrr"]
            if new > cur + 1e-9:
                cur = new
            else:
                a[i] = 1 - a[i]
    return a


def loo_gains(env: Env) -> torch.Tensor:
    """조문별 leave-one-out 이득 (학습 질문 MRR 기준).

    g_i = 0.5 * [R(i만 on) - R(전부 off)] + 0.5 * [R(전부 on) - R(i만 off)]
    두 기준점에서 평균내서 다른 조문 설정에 대한 의존을 줄인다.
    """
    n = len(env.act)
    off, on = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    base_off = env.evaluate(off, env.train)["mrr"]
    base_on = env.evaluate(on, env.train)["mrr"]
    g = torch.zeros(n, device=DEV)
    for i in range(n):
        a = off.clone(); a[i] = 1
        b = on.clone(); b[i] = 0
        g[i] = 0.5 * (env.evaluate(a, env.train)["mrr"] - base_off) + 0.5 * (base_on - env.evaluate(b, env.train)["mrr"])
    return g


def fit_ridge(X, y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1], device=DEV)
    return torch.linalg.solve(A, X.T @ y)


def main_loo(law="민법"):
    env = Env(law)
    n = len(env.act)
    g = loo_gains(env)
    is_test = torch.tensor([env.ids[k] in env.test_art for k in env.act], device=DEV)
    print(f"LOO 이득: 평균 {g.mean():.5f}, 양수 {int((g > 0).sum())}/{n}, 최대 {g.max():.4f}, 최소 {g.min():.4f}")
    tr = ~is_test
    rows = {
        "전부 off": torch.zeros(n, device=DEV),
        "전부 on": torch.ones(n, device=DEV),
    }
    for name, X in (("기본 특징", env.feat), ("보강 특징", env.feat2)):
        for lam in (1.0, 10.0, 100.0):
            w = fit_ridge(X[tr], g[tr], lam)
            rows[f"LOO+ridge {name} λ={lam:g}"] = ((X @ w) > 0).float()
    print(f"{'정책':<30}{'on 비율':>8}{'학습MRR':>9}{'테스트MRR':>10}{'테스트R@10':>11}{'멀티MRR':>9}{'단일MRR':>9}")
    for name, a in rows.items():
        t, e = env.evaluate(a, env.train), env.evaluate(a, env.test)
        print(f"{name:<30}{a.mean().item():>8.2f}{t['mrr']:>9.3f}{e['mrr']:>10.3f}{e['r10']:>11.3f}{e['mrr_multi']:>9.3f}{e['mrr_single']:>9.3f}")
    # 이득 예측력: 테스트 조문에서의 실제 이득(테스트 질문 기준)과 예측의 상관
    gt = torch.zeros(n, device=DEV)
    off, on = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    bo, bn = env.evaluate(off, env.test)["mrr"], env.evaluate(on, env.test)["mrr"]
    for i in range(n):
        a = off.clone(); a[i] = 1
        b = on.clone(); b[i] = 0
        gt[i] = 0.5 * (env.evaluate(a, env.test)["mrr"] - bo) + 0.5 * (bn - env.evaluate(b, env.test)["mrr"])
    for name, X in (("기본", env.feat), ("보강", env.feat2)):
        w = fit_ridge(X[tr], g[tr], 10.0)
        pred = (X @ w)[is_test]
        c = torch.corrcoef(torch.stack([pred, gt[is_test]]))[0, 1].item()
        print(f"{name} 특징: 테스트 조문 {int(is_test.sum())}개에서 예측 이득 vs 실제 이득 상관 {c:.3f}")


def main(law="민법", seeds=5):
    env = Env(law)
    n = len(env.act)
    rows = {}
    g = torch.Generator(device=DEV).manual_seed(123)
    rows["전부 off (article)"] = torch.zeros(n, device=DEV)
    rows["전부 on (article+refs)"] = torch.ones(n, device=DEV)
    rows["무작위 50%"] = (torch.rand(n, device=DEV, generator=g) < 0.5).float()
    rows["오라클(테스트 질문에 튜닝, 상한)"] = oracle(env, env.test)
    learned = []
    for s in range(seeds):
        theta = train(env, seed=s)
        learned.append((torch.sigmoid(env.feat @ theta) > 0.5).float())
    print(f"{'정책':<34}{'on 비율':>8}{'학습MRR':>9}{'테스트MRR':>10}{'테스트R@10':>11}{'멀티MRR':>9}{'단일MRR':>9}")
    def show(name, a, tag=""):
        tr, te = env.evaluate(a, env.train), env.evaluate(a, env.test)
        print(f"{name:<34}{a.mean().item():>8.2f}{tr['mrr']:>9.3f}{te['mrr']:>10.3f}{te['r10']:>11.3f}{te['mrr_multi']:>9.3f}{te['mrr_single']:>9.3f}")
        return te["mrr"]
    for k, a in rows.items():
        show(k, a)
    ms = [show(f"학습 정책 seed{s}", a) for s, a in enumerate(learned)]
    print(f"학습 정책 테스트 MRR 평균 {np.mean(ms):.3f} ± {np.std(ms):.3f} (seed {seeds}개)")


if __name__ == "__main__":
    law = sys.argv[1] if len(sys.argv) > 1 else "민법"
    (main_loo if "loo" in sys.argv[2:] else main)(law)
