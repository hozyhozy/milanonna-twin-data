#!/usr/bin/env python3
"""⑥ 같은 추출기로 비교: 실제 답과 트윈 답을 똑같은 추출기로 분해해 칸별로 맞대어 본다.
Etudo & Yoon에서 조직 문서와 새 글을 같은 추출기로 뽑아 관계별로 비교한 것(FRP)에 해당.

.venv-st/bin/python experiments/ontology/compare.py --run pilot4 --clean --ceiling
.venv-st/bin/python experiments/ontology/compare.py --run pilot3 --rescore          # LLM 호출 없이 지표만 다시 계산

지표 (v3)
  topic_match  실제 답의 입장 중 트윈 답에 같은 대상의 입장이 있는 비율
  polarity     대상이 맞은 입장끼리 찬반이 같은 비율
  attr / cons  특징·결과 꼬리표 겹침 (자카드) — 꼬리표를 합친(코드북) 뒤에야 의미가 있음
  value        Schwartz 칸 겹침 (자카드). 정답 쪽 가치가 '직접 진술'인 문항과 '추론뿐'인 문항을 나눠서도 보고
  ep_reused    트윈이 든 일화 중 Train에 있는 본인 일화 (가져다 씀)
  ep_invented  트윈이 든 일화 중 Train 어디에도 없는 일화 (지어냈을 가능성)
  char_sim     문자 n-gram 코사인 (참고용 표면 유사도)

v3 (2026-11-23)
  - 대상·일화 맞추기: e5 임베딩 코사인(--match emb, 기본). 기준값은 고정하지 않고 자료에서 정한다:
    서로 다른 문항의 대상끼리(무관한 쌍) 코사인의 95번째 백분위수보다 가까우면 '같은 대상'.
    일화도 같은 방식(Train 일화끼리 무관한 쌍의 95번째 백분위수). 예전 방식은 --match char (문자 유사도 ≥ 0.3).
  - 추출과 채점을 나눔: 추출 결과(truth_ext, twin_ext)를 compare.jsonl에 쌓고, 채점은 매번 전체를 다시 한다.
    그래서 --rescore 로 LLM 호출 없이 지표만 다시 계산할 수 있다.
  - --reuse-truth-from RUN: 다른 run의 정답 추출과 천장 행을 그대로 가져온다(같은 정답을 다시 뽑지 않아 비교가 공정).
  - 조건 간 차이: 같은 문항끼리 짝지어 평균 차이와 95% 부트스트랩 구간을 compare.md에 붙인다.
--ceiling: Retest(같은 질문에 본인이 다른 때 한 답) 쌍으로 같은 지표를 계산 → '천장' 행.
"""
import argparse, json, os, random
from collections import defaultdict, Counter
import numpy as np
from common import run_dir, read_jsonl, TagBook, load_split, load_clean

CHAR_TOPIC, CHAR_EP = 0.3, 0.3
METRICS = ["topic_match", "polarity", "attr", "cons", "value", "ep_reused", "ep_invented", "char_sim"]


def char_sim(a, b):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not a or not b:
        return 0.0
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit([a, b])
    M = v.transform([a, b])
    return float((M[0] @ M[1].T).toarray()[0][0])


class Matcher:
    """대상·일화가 '같은지' 판정. emb: e5 코사인 + 자료에서 정한 기준값, char: 문자 유사도 고정 기준값."""

    def __init__(self, how, topics_by_item, train_eps):
        self.how = how
        if how == "emb":
            from embed import Embedder, require_st
            if not require_st("compare.py 대상 맞추기"):
                self.how = "char"
        if self.how == "emb":
            self.E = Embedder("st")
            self.t_topic = self._calib([(i, t) for i, ts in topics_by_item.items() for t in ts])
            self.t_ep = self._calib(list(enumerate(train_eps)))
            self.train_eps = train_eps
            self.EP = self.E.encode(train_eps) if train_eps else None
        else:
            self.t_topic, self.t_ep, self.train_eps = CHAR_TOPIC, CHAR_EP, train_eps

    def _calib(self, pairs_src, n=3000, q=95):
        """서로 다른 묶음(문항)에서 뽑은 무관한 쌍의 코사인 q번째 백분위수."""
        rng = random.Random(0)
        if len(pairs_src) < 10:
            return 0.9
        texts = list({t for _, t in pairs_src})
        V = dict(zip(texts, self.E.encode(texts)))
        sims = []
        for _ in range(n):
            (i, a), (j, b) = rng.sample(pairs_src, 2)
            if i != j and a != b:
                sims.append(float(V[a] @ V[b]))
        return float(np.percentile(sims, q)) if sims else 0.9

    def best(self, a_list, b_list):
        """a의 각 항목에 대해 b 중 가장 가까운 것의 (위치, 유사도)."""
        if not a_list or not b_list:
            return [(None, 0.0)] * len(a_list)
        if self.how == "emb":
            S = self.E.encode(a_list) @ self.E.encode(b_list).T
            return [(int(r.argmax()), float(r.max())) for r in S]
        out = []
        for a in a_list:
            s = [char_sim(a, b) for b in b_list]
            out.append((int(np.argmax(s)), float(max(s))))
        return out

    def ep_known(self, texts):
        if not texts or not self.train_eps:
            return [False] * len(texts)
        if self.how == "emb":
            S = self.E.encode(texts) @ self.EP.T
            return [float(r.max()) >= self.t_ep for r in S]
        return [max(char_sim(t, x) for x in self.train_eps) >= self.t_ep for t in texts]


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else None


def schwartz(ext):
    return [y for x in ext["stances"] if x.get("value") for y in x["value"].get("schwartz", []) if y != "해당 없음"]


def value_direct(ext):
    return any(str((x.get("value") or {}).get("evidence", "")).startswith("직접") for x in ext["stances"])


def score(te, ae, M, truth_text, twin_text):
    T, A = te["stances"], ae["stances"]
    matched = agree = 0
    for t, (j, s) in zip(T, M.best([t["topic"] for t in T], [x["topic"] for x in A])):
        if j is not None and s >= M.t_topic:
            matched += 1
            agree += (t["polarity"] == A[j]["polarity"])
    known = M.ep_known([f"{e['label']} {e['phrase']}" for e in ae["episodes"]])
    return {"topic_match": matched / len(T) if T else None,
            "polarity": agree / matched if matched else None,
            "attr": jac([x["attribute"]["tag"] for x in T if x.get("attribute")], [x["attribute"]["tag"] for x in A if x.get("attribute")]),
            "cons": jac([x["consequence"]["tag"] for x in T if x.get("consequence")], [x["consequence"]["tag"] for x in A if x.get("consequence")]),
            "value": jac(schwartz(te), schwartz(ae)),
            "truth_value_direct": value_direct(te),
            "ep_reused": sum(known), "ep_invented": len(known) - sum(known), "ep_truth": len(te["episodes"]),
            "char_sim": char_sim(truth_text, twin_text)}


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(B))
    return bs[int(B * 0.025)], bs[int(B * 0.975) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot2")
    ap.add_argument("--clean", action="store_true", help="실제 답도 교정본으로 (있을 때)")
    ap.add_argument("--ceiling", action="store_true", help="Retest 쌍으로 천장 계산")
    ap.add_argument("--match", default="emb", choices=["emb", "char"], help="대상·일화 맞추기 방식")
    ap.add_argument("--reuse-truth-from", default="", help="정답 추출·천장 행을 가져올 run (예: pilot3)")
    ap.add_argument("--rescore", action="store_true", help="LLM 호출 없이 쌓인 추출로 지표만 다시 계산")
    a = ap.parse_args()
    d = run_dir(a.run)
    g = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    train_eps = [f"{b['label'].split(' · ', 1)[-1]} {b['phrase']}" for b in g["boxes"] if b["type"] == "Episode"]
    clean = load_clean() if a.clean else {}

    # 비교할 쌍: (문항, 조건) → (질문, 정답 원문, 트윈 원문)
    jobs = {}
    for r in read_jsonl(os.path.join(d, "answers.jsonl")):
        jobs[(r["item_id"], r["cond"])] = (r["question"], clean.get(r["item_id"], r["truth"]), r["answer"])
    if a.ceiling:                           # 본인 재답변: Retest 답 = 실제, 같은 묶음의 Train 답 = '트윈'
        groups = defaultdict(list)
        for x in load_split("Train"):
            if x.get("dup_group"):
                groups[x["dup_group"]].append(x)
        for x in load_split("Retest"):
            for y in groups.get(x.get("dup_group"), [])[:1]:
                jobs[(x["id"], "천장")] = (x["question"], clean.get(x["id"], x["answer"]), clean.get(y["id"], y["answer"]))

    cache_p = os.path.join(d, "compare.jsonl")
    recs = {(r["item_id"], r["cond"]): r for r in read_jsonl(cache_p) if "twin_ext" in r}
    reuse_truth, reuse_ceiling = {}, {}
    if a.reuse_truth_from:
        for r in read_jsonl(os.path.join(run_dir(a.reuse_truth_from), "compare.jsonl")):
            if "truth_ext" in r:
                reuse_truth.setdefault(r["item_id"], r["truth_ext"])
                if r["cond"] == "천장":
                    reuse_ceiling[r["item_id"]] = r

    if not a.rescore:
        from extract import extract_one, get_llm
        tb = TagBook()
        tp = os.path.join(d, "tags.json")
        tags = json.load(open(tp, encoding="utf-8")) if os.path.exists(tp) else {}
        for k in tb.tags:
            tb.tags[k] = tags.get(k, {})      # 꼬리표 목록은 고정 (자라지 않음)
        llm = None
        truth_ext = {item: r["truth_ext"] for (item, _), r in recs.items()}
        truth_ext.update({k: v for k, v in reuse_truth.items() if k not in truth_ext})
        with open(cache_p, "a", encoding="utf-8") as f:
            for (item, cond), (q, truth, twin) in jobs.items():
                if (item, cond) in recs:
                    continue
                if cond == "천장" and item in reuse_ceiling:
                    rec = {k: reuse_ceiling[item][k] for k in ("item_id", "cond", "truth_ext", "twin_ext")}
                else:
                    llm = llm or get_llm()
                    if item not in truth_ext:
                        truth_ext[item] = extract_one(llm, q, truth, tb)[0]
                    rec = {"item_id": item, "cond": cond, "truth_ext": truth_ext[item],
                           "twin_ext": extract_one(llm, q, twin, tb)[0]}
                recs[(item, cond)] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print("추출", item, cond)

    # ---- 채점 (매번 전체) ----
    keys = [k for k in recs if k in jobs]
    topics_by_item = defaultdict(list)
    for k in keys:
        for side in ("truth_ext", "twin_ext"):
            topics_by_item[k[0]] += [s["topic"] for s in recs[k][side]["stances"] if s.get("topic")]
    M = Matcher(a.match, topics_by_item, train_eps)
    scored = {}
    for k in keys:
        q, truth, twin = jobs[k]
        r = recs[k]
        scored[k] = {"item_id": k[0], "cond": k[1], **score(r["truth_ext"], r["twin_ext"], M, truth, twin),
                     "truth_ext": r["truth_ext"], "twin_ext": r["twin_ext"]}
    with open(cache_p, "w", encoding="utf-8") as f:
        for r in scored.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 기준선: 질문과 상관없이 '이 사람이 가장 자주 지키는 가치 2개'를 늘 답했다고 볼 때의 가치 겹침
    top2 = [k for k, _ in Counter(x for b in g["boxes"] if b["type"] == "Stance" for x in b.get("schwartz", [])
                                  if x != "해당 없음").most_common(2)]
    agg = defaultdict(lambda: defaultdict(list))
    for (item, cond), r in scored.items():
        if cond == "B0":
            j = jac(schwartz(r["truth_ext"]), top2)
            if j is not None:
                agg["최빈 가치"]["value"].append(j)
                agg["최빈 가치"]["value_d" if r["truth_value_direct"] else "value_i"].append(j)
    for r in scored.values():
        for m in METRICS:
            if r.get(m) is not None:
                agg[r["cond"]][m].append(float(r[m]))
        if r.get("value") is not None:
            agg[r["cond"]]["value_d" if r["truth_value_direct"] else "value_i"].append(float(r["value"]))
    mean = lambda xs: f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "—"
    how = (f"e5 코사인, 대상 기준 {M.t_topic:.3f} · 일화 기준 {M.t_ep:.3f} (무관한 쌍의 95번째 백분위수)"
           if M.how == "emb" else f"문자 유사도, 대상·일화 기준 {M.t_topic}")
    L = ["# 같은 추출기로 비교한 충실도 프로필 (v3 지표)", "", f"맞추기: {how}", "",
         "| 조건 | 대상 맞음 | 찬반 일치(맞은 대상 중) | 특징 겹침 | 결과 겹침 | 가치 겹침 | 가치 겹침 (정답 직접 진술) | 가치 겹침 (정답 추론뿐) | 본인 일화 재사용 | 근거 없는 일화 | 표면 유사도 |",
         "|" + "---|" * 12]
    order = [c for c in ["천장", "O1", "B1", "B0", "최빈 가치"] if c in agg] + sorted(c for c in agg if c not in ("천장", "O1", "B1", "B0", "최빈 가치"))
    for c in order:
        m = agg[c]
        L.append(f"| {c} | {mean(m['topic_match'])} | {mean(m['polarity'])} | {mean(m['attr'])} | {mean(m['cons'])} | "
                 f"{mean(m['value'])} | {mean(m['value_d'])} | {mean(m['value_i'])} | {mean(m['ep_reused'])} | {mean(m['ep_invented'])} | {mean(m['char_sim'])} |")

    # 조건 간 차이 (같은 문항끼리)
    by = defaultdict(dict)
    for (item, cond), r in scored.items():
        by[cond][item] = r
    L += ["", "## 조건 간 차이 (같은 문항끼리 짝지음, 평균 차이와 95% 부트스트랩 구간, * = 구간이 0을 넘지 않음)", "",
          "| 지표 | O1 − B1 | O1 − B0 | B1 − B0 |", "|---|---|---|---|"]
    names = {"topic_match": "대상 맞음", "attr": "특징 겹침", "cons": "결과 겹침", "value": "가치 겹침",
             "ep_reused": "본인 일화 재사용", "ep_invented": "근거 없는 일화"}
    for m, nm in names.items():
        cells = []
        for x, y in [("O1", "B1"), ("O1", "B0"), ("B1", "B0")]:
            dd = [by[x][i][m] - by[y][i][m] for i in by.get(x, {}) if i in by.get(y, {})
                  and by[x][i].get(m) is not None and by[y][i].get(m) is not None]
            if len(dd) < 5:
                cells.append("—"); continue
            lo, hi = boot(dd)
            star = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{sum(dd) / len(dd):+.2f} ({lo:+.2f} ~ {hi:+.2f}){star} n={len(dd)}")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")

    L += ["", f"최빈 가치 = 질문과 상관없이 늘 {', '.join(top2)}라고 답했다고 칠 때의 가치 겹침. 트윈의 가치 겹침이 이보다 높아야 의미가 있음.",
          "가치 겹침 (정답 직접 진술) = 실제 답에서 본인이 가치를 직접 말한 문항만. 추론뿐인 문항은 정답 쪽 가치부터 흔들리므로 따로 봄.",
          "천장 = 같은 질문에 본인이 다른 때 한 답끼리 비교(Retest). 트윈이 천장에 가까울수록 좋음.",
          "B0 = 기본 소개만, B1 = 원문 발췌, O1 = 관계도 근거.",
          "특징·결과 겹침은 꼬리표를 합친(코드북) 뒤에 의미가 있습니다. 근거 없는 일화는 자동 판정이니 몇 개는 손으로 확인하세요."]
    open(os.path.join(d, "compare.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
