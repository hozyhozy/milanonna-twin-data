#!/usr/bin/env python3
"""⑥ 같은 추출기로 비교: 실제 답과 트윈 답을 똑같은 추출기로 분해해 칸별로 맞대어 본다.
Etudo & Yoon에서 조직 문서와 새 글을 같은 추출기로 뽑아 관계별로 비교한 것(FRP)에 해당.

python3 experiments/ontology/compare.py --run pilot2 [--clean] [--ceiling]

지표 (v2)
  topic_match  실제 답의 입장 중 트윈 답에 같은 대상의 입장이 있는 비율 (대상 문자 유사도 ≥ 0.3)
  polarity     대상이 맞은 입장끼리 찬반이 같은 비율
  attr         특징 꼬리표 겹침 (자카드) — 꼬리표를 합친(코드북) 뒤에야 의미가 있음
  cons         결과 꼬리표 겹침 (자카드) — 설계표 v0.4부터 C도 공유 개념
  value        Schwartz 칸 겹침 (자카드)
  ep_reused    트윈이 든 일화 중 Train에 있는 본인 일화 (가져다 씀)
  ep_invented  트윈이 든 일화 중 Train 어디에도 없는 일화 (지어냈을 가능성)
  char_sim     문자 n-gram 코사인 (참고용 표면 유사도)
--ceiling: Retest(같은 질문에 본인이 다른 때 한 답) 쌍으로 같은 지표를 계산 → '천장' 행.
"""
import argparse, json, os
from collections import defaultdict
from common import run_dir, read_jsonl, TagBook, load_split, load_clean
from extract import extract_one, get_llm

TOPIC_SIM, EP_SIM = 0.3, 0.3


def make_sim():
    from sklearn.feature_extraction.text import TfidfVectorizer
    def sim(a, b):
        if not a or not b:
            return 0.0
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit([a, b])
        M = v.transform([a, b])
        return float((M[0] @ M[1].T).toarray()[0][0])
    return sim


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else None


def score(te, ae, sim, train_eps, truth_text, twin_text):
    T, A = te["stances"], ae["stances"]
    matched, agree = 0, 0
    for t in T:
        best = max(A, key=lambda x: sim(t["topic"], x["topic"]), default=None)
        if best is not None and sim(t["topic"], best["topic"]) >= TOPIC_SIM:
            matched += 1
            agree += (t["polarity"] == best["polarity"])
    reused = invented = 0
    for e in ae["episodes"]:
        q = f"{e['label']} {e['phrase']}"
        if train_eps and max(sim(q, x) for x in train_eps) >= EP_SIM:
            reused += 1
        else:
            invented += 1
    return {"topic_match": matched / len(T) if T else None,
            "polarity": agree / matched if matched else None,
            "attr": jac([x["attribute"]["tag"] for x in T if x["attribute"]], [x["attribute"]["tag"] for x in A if x["attribute"]]),
            "cons": jac([x["consequence"]["tag"] for x in T if x["consequence"]], [x["consequence"]["tag"] for x in A if x["consequence"]]),
            "value": jac([y for x in T if x["value"] for y in x["value"]["schwartz"] if y != "해당 없음"],
                         [y for x in A if x["value"] for y in x["value"]["schwartz"] if y != "해당 없음"]),
            "ep_reused": reused, "ep_invented": invented, "ep_truth": len(te["episodes"]),
            "char_sim": sim(truth_text, twin_text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot2")
    ap.add_argument("--clean", action="store_true", help="실제 답도 교정본으로 (있을 때)")
    ap.add_argument("--ceiling", action="store_true", help="Retest 쌍으로 천장 계산")
    a = ap.parse_args()
    d = run_dir(a.run)
    tb = TagBook()
    tags = json.load(open(os.path.join(d, "tags.json"), encoding="utf-8"))
    for k in tb.tags:
        tb.tags[k] = tags.get(k, {})          # 꼬리표 목록은 고정 (자라지 않음)
    g = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    train_eps = [f"{b['label'].split(' · ', 1)[-1]} {b['phrase']}" for b in g["boxes"] if b["type"] == "Episode"]
    llm, sim = get_llm(), make_sim()
    clean = load_clean() if a.clean else {}
    cache_p = os.path.join(d, "compare.jsonl")
    done = {(r["item_id"], r["cond"]): r for r in read_jsonl(cache_p) if "topic_match" in r}
    truth_ext = {}
    jobs = [(r["item_id"], r["cond"], r["question"], clean.get(r["item_id"], r["truth"]), r["answer"])
            for r in read_jsonl(os.path.join(d, "answers.jsonl"))]
    if a.ceiling:                           # 본인 재답변: Retest 답 = 실제, 같은 묶음의 Train 답 = '트윈'
        train = {x["id"]: x for x in load_split("Train")}
        groups = defaultdict(list)
        for x in train.values():
            if x.get("dup_group"):
                groups[x["dup_group"]].append(x)
        for x in load_split("Retest"):
            for y in groups.get(x.get("dup_group"), [])[:1]:
                jobs.append((x["id"], "천장", x["question"], clean.get(x["id"], x["answer"]), clean.get(y["id"], y["answer"])))
    with open(cache_p, "a", encoding="utf-8") as f:
        for item, cond, q, truth, twin in jobs:
            if (item, cond) in done:
                continue
            if (item, q) not in truth_ext:
                truth_ext[(item, q)] = extract_one(llm, q, truth, tb)[0]
            te = truth_ext[(item, q)]
            ae = extract_one(llm, q, twin, tb)[0]
            rec = {"item_id": item, "cond": cond, **score(te, ae, sim, train_eps, truth, twin),
                   "truth_ext": te, "twin_ext": ae}
            done[(item, cond)] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(item, cond, rec["topic_match"], rec["polarity"], rec["value"], rec["ep_invented"])
    # 기준선: 질문과 상관없이 '이 사람이 가장 자주 지키는 가치 2개'를 늘 답했다고 볼 때의 가치 겹침
    from collections import Counter
    top2 = [k for k, _ in Counter(x for b in g["boxes"] if b["type"] == "Stance" for x in b.get("schwartz", [])
                                  if x != "해당 없음").most_common(2)]
    base = defaultdict(lambda: defaultdict(list))
    for (item, cond), r in done.items():
        if cond == "B0":
            tv = [y for x in r["truth_ext"]["stances"] if x["value"] for y in x["value"]["schwartz"] if y != "해당 없음"]
            j = jac(tv, top2)
            if j is not None:
                base["최빈 가치"]["value"].append(j)
    agg = defaultdict(lambda: defaultdict(list))
    agg.update(base)
    for r in done.values():
        for k in ["topic_match", "polarity", "attr", "cons", "value", "ep_reused", "ep_invented", "char_sim"]:
            if r.get(k) is not None:
                agg[r["cond"]][k].append(float(r[k]))
    mean = lambda xs: f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "—"
    L = ["# 같은 추출기로 비교한 충실도 프로필 (v2 지표)", "",
         "| 조건 | 대상 맞음 | 찬반 일치(맞은 대상 중) | 특징 겹침 | 결과 겹침 | 가치 겹침 | 본인 일화 재사용 | 근거 없는 일화 | 표면 유사도 |",
         "|" + "---|" * 10]
    order = [c for c in ["천장", "O1", "B1", "B0", "최빈 가치"] if c in agg] + [c for c in sorted(agg) if c not in ("천장", "O1", "B1", "B0", "최빈 가치")]
    for c in order:
        m = agg[c]
        L.append(f"| {c} | {mean(m['topic_match'])} | {mean(m['polarity'])} | {mean(m['attr'])} | {mean(m['cons'])} | {mean(m['value'])} | "
                 f"{mean(m['ep_reused'])} | {mean(m['ep_invented'])} | {mean(m['char_sim'])} |")
    L += ["", f"최빈 가치 = 질문과 상관없이 늘 {', '.join(top2)}라고 답했다고 칠 때의 가치 겹침. 트윈의 가치 겹침이 이보다 높아야 의미가 있음.",
          "천장 = 같은 질문에 본인이 다른 때 한 답끼리 비교(Retest). 트윈이 천장에 가까울수록 좋음. 단, 천장의 일화는 관계도에 들어간 발화가 적으면 '근거 없음'으로 잘못 잡힘.",
          "B0 = 기본 소개만, B1 = 원문 발췌, O1 = 관계도 근거.",
          "특징·결과 겹침은 꼬리표를 합친(코드북) 뒤에 의미가 있습니다. 근거 없는 일화는 Train 일화와 문자 유사도로만 판정하니 손으로 확인이 필요합니다."]
    open(os.path.join(d, "compare.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
