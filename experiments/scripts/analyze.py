#!/usr/bin/env python3
"""⑤-3 집계. judge_*.jsonl + runs 를 읽어 results/<exp>_summary.md / .csv 를 만든다.

표 1  조건별 agree(stance/ground/valence/voice)·sim_emb 평균 (+ Retest 천장 대비 비율)
표 2  층별(stratum)·유명/신규(fame)·엄격(strict) 분해
표 3  거리 삼각형: dist(G2–본인) vs dist(G2–G0/G1/G3) — 같은 문항·같은 rep의 응답 간 임베딩 거리
표 4  A3 분산: 조건·온도별 20회 응답 간 상호 거리 ÷ Retest 거리
표 5  A2 valence_gap, C1 근거 유형 분포·환각 후보 (judge --tag 실행 시)
표 6  B1 충돌/일치 집합(G0 stance 하위/상위 1/3)에서 G2 일치도와 끌림 지수
표 7  E1 leak_index = G3−G0, 순기여 = G2−G3 (유명/신규별)
표 8  F2 acc (돌렸을 때만)

python3 experiments/scripts/analyze.py --exp A1
"""
import json, os, sys, argparse, glob, collections, statistics as st, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import Embedder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = lambda *a: os.path.join(ROOT, *a)
KEYS = ["stance", "ground", "valence", "voice"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 2) if xs else None


def item_mean(js, key_fn):
    """문항별 평균을 먼저 낸 뒤 문항 간 평균 (문항마다 반복 수가 달라도 동일 가중)."""
    by = collections.defaultdict(list)
    for j in js:
        v = key_fn(j)
        if v is not None:
            by[j["item_id"]].append(v)
    return mean([st.mean(v) for v in by.values()]) if by else None


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="A1"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--max-rep", type=int, default=5, help="T>0 반복은 rep < max_rep 만 사용(조건 간 반복 수 균형)")
    a = ap.parse_args()
    items = {x["item_id"]: x for x in json.load(open(P("experiments/items/holdout50.json"), encoding="utf-8"))["items"]}
    J, R = [], []
    for base in ["experiments/runs", "experiments/runs_local"]:
        for f in glob.glob(P(base, a.exp, "*.jsonl")):
            (J if os.path.basename(f).startswith("judge_") else R).extend(load_jsonl(f))
    if not J:
        print("judge 결과가 없습니다. judge.py를 먼저 실행하세요."); return
    # 조건 간 반복 수를 맞춘다 (한 조건만 더 많이 돌아간 경우 평균이 문항 구성에 끌리는 것을 방지)
    J = [j for j in J if j["temp"] == 0 or j["rep"] < a.max_rep]
    R = [r for r in R if r["temp"] == 0 or r["rep"] < a.max_rep]
    conds = sorted({j["cond"] for j in J})
    out = [f"# {a.exp} 집계 ({len(J)} 채점, 조건 {', '.join(conds)}, T>0 반복 rep<{a.max_rep}, 문항 균등 가중)\n"]
    rows = []

    # 천장
    ceil_p = P("experiments/results", f"{a.exp}_retest_ceiling.json")
    ceil = None
    if os.path.exists(ceil_p):
        c = [x for x in json.load(open(ceil_p, encoding="utf-8")) if x["scores"]]
        if c:
            ceil = {"stance": mean([x["scores"]["stance"] for x in c]), "sim": mean([x["sim_emb"] for x in c]),
                    "dist": round(1 - st.mean([x["sim_emb"] for x in c]), 4)}

    # 표1
    out.append("## 표1. 조건별 일치도 (F1, T>0 평균)\n\n| 조건 | n | stance | ground | valence | voice | sim_emb | stance/천장 |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in conds:
        js = [j for j in J if j["cond"] == c and j["temp"] > 0 and j["scores"]]
        m = {k: item_mean(js, lambda j, k=k: j["scores"][k]) for k in KEYS}
        se = item_mean(js, lambda j: j["sim_emb"])
        ratio = round(m["stance"] / ceil["stance"], 2) if ceil and m["stance"] else ""
        out.append(f"| {c} | {len(js)} | {m['stance']} | {m['ground']} | {m['valence']} | {m['voice']} | {se} | {ratio} |")
        rows.append({"table": "cond", "cond": c, "n": len(js), **m, "sim_emb": se})
    if ceil:
        out.append(f"\n본인 Retest 천장: stance {ceil['stance']}, sim_emb {ceil['sim']}")
    # 판정 자기 일관성
    sw = [j for j in J if j.get("scores_swap") and j["scores"]]
    if sw:
        import math
        xs = [j["scores"]["stance"] for j in sw]; ys = [j["scores_swap"]["stance"] for j in sw]
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) or 1
        out.append(f"\n판정 자기 일관성(A/B 바꿔 재채점, stance 상관): r = {num/den:.3f} (n={len(sw)})")

    # 표2
    out.append("\n## 표2. 층·유명·엄격별 stance 일치도\n\n| 구분 | 값 | " + " | ".join(conds) + " |\n|---|---|" + "---:|" * len(conds))
    for key in ["stratum", "fame", "strict"]:
        vals = sorted({str(items[j["item_id"]].get(key)) for j in J})
        for v in vals:
            cells = []
            for c in conds:
                js = [j for j in J if j["cond"] == c and j["temp"] > 0 and j["scores"] and str(items[j["item_id"]].get(key)) == v]
                cells.append(str(item_mean(js, lambda j: j["scores"]["stance"])))
            out.append(f"| {key} | {v} | " + " | ".join(cells) + " |")

    # 표3 거리 삼각형 (응답 텍스트 필요)
    resp = {(r["item_id"], r["cond"], r["temp"], r["rep"]): r["response"] for r in R if r["format"] == "F1"}
    if resp and "G2" in conds:
        emb = Embedder(a.backend, fit_corpus=list(resp.values()) + [x["answer_truth"] for x in items.values()])
        pairs = collections.defaultdict(list)
        keys2 = [k for k in resp if k[1] == "G2" and k[2] > 0]
        A, B, lab = [], [], []
        for (iid, _, t, rep) in keys2:
            A.append(resp[(iid, "G2", t, rep)]); B.append(items[iid]["answer_truth"]); lab.append("본인")
            for oc in [c for c in conds if c != "G2"]:
                k2 = (iid, oc, t, rep)
                if k2 in resp:
                    A.append(resp[(iid, "G2", t, rep)]); B.append(resp[k2]); lab.append(oc)
        sims = emb.sim(A, B)
        for l, s in zip(lab, sims):
            pairs[l].append(1 - s)
        out.append("\n## 표3. 거리 삼각형 — G2 응답과의 평균 거리(1−sim_emb, 작을수록 가까움)\n\n| 상대 | n | dist |\n|---|---:|---:|")
        for l, v in pairs.items():
            out.append(f"| G2–{l} | {len(v)} | {round(st.mean(v), 4)} |")
        out.append("\n해석: dist(G2–G0) < dist(G2–본인) → 개인화 부족 / dist(G2–G1) < dist(G2–본인) → 전형 의존 / dist(G2–G3) < dist(G2–본인) → 사전지식 지배")

        # 표4 분산
        out.append("\n## 표4. 응답 간 상호 거리 (문항별 반복 응답끼리, 평균)\n\n| 조건 | temp | n문항 | 상호 거리 | ÷Retest 거리 |\n|---|---:|---:|---:|---:|")
        for c in conds:
            for t in sorted({k[2] for k in resp if k[1] == c and k[2] > 0}):
                ds = []
                for iid in items:
                    rs = [resp[k] for k in resp if k[0] == iid and k[1] == c and k[2] == t]
                    if len(rs) >= 2:
                        pa, pb = zip(*itertools.combinations(rs, 2))
                        ds.append(st.mean([1 - s for s in emb.sim(list(pa), list(pb))]))
                if ds:
                    ratio = round(st.mean(ds) / ceil["dist"], 2) if ceil else ""
                    out.append(f"| {c} | {t} | {len(ds)} | {round(st.mean(ds), 4)} | {ratio} |")

    # 표5 A2/C1
    tagged = [j for j in J if j.get("resp_tags")]
    if tagged:
        out.append("\n## 표5. A2 valence_gap(응답 − 본인) · C1 근거 유형\n\n| 조건 | n | valence_gap | 부정문항 gap | 긍정문항 gap | 일화% | 원칙% | 사실% | 혼합% | 응답당 일화 수 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in conds:
            js = [j for j in tagged if j["cond"] == c and j["temp"] > 0]
            gaps = [(j["resp_tags"].get("valence") or 0) - (items[j["item_id"]].get("valence_truth") or 0) for j in js if items[j["item_id"]].get("valence_truth") is not None]
            neg = [(j["resp_tags"].get("valence") or 0) - items[j["item_id"]]["valence_truth"] for j in js if (items[j["item_id"]].get("valence_truth") or 0) <= -1]
            pos = [(j["resp_tags"].get("valence") or 0) - items[j["item_id"]]["valence_truth"] for j in js if (items[j["item_id"]].get("valence_truth") or 0) >= 1]
            g = collections.Counter(j["resp_tags"].get("ground") for j in js)
            tot = max(1, len(js))
            an = mean([len(j["resp_tags"].get("anecdotes") or []) for j in js])
            out.append(f"| {c} | {len(js)} | {mean(gaps)} | {mean(neg)} | {mean(pos)} | {round(100*g['일화']/tot)} | {round(100*g['원칙']/tot)} | {round(100*g['사실']/tot)} | {round(100*g['혼합']/tot)} | {an} |")
        tg = collections.Counter(x.get("ground_type_truth") for x in items.values())
        out.append(f"\n본인 근거 유형 분포: {dict(tg)}. 환각 후보는 응답 속 일화 목록을 Train·G1과 대조해 연구자가 확인(`judge_*.jsonl`의 resp_tags.anecdotes).")

    # 표6 B1
    if "G0" in conds and "G2" in conds:
        g0 = {}
        for j in J:
            if j["cond"] == "G0" and j["temp"] == 0 and j["scores"]:
                g0[j["item_id"]] = j["scores"]["stance"]
        if len(g0) >= 6:
            srt = sorted(g0, key=g0.get); k = len(srt) // 3
            conflict, agree = set(srt[:k]), set(srt[-k:])
            out.append("\n## 표6. B1 가치 충돌 — G0 T=0 stance 하위 1/3(충돌) vs 상위 1/3(일치)\n\n| 집합 | n문항 | G0 stance | G2 stance | 끌림 지수 |\n|---|---:|---:|---:|---:|")
            for name, S in [("충돌", conflict), ("일치", agree)]:
                js2 = [j for j in J if j["cond"] == "G2" and j["temp"] > 0 and j["scores"] and j["item_id"] in S]
                js0 = [j for j in J if j["cond"] == "G0" and j["temp"] > 0 and j["scores"] and j["item_id"] in S]
                pull = ""
                if resp:
                    d_self, d_g0 = [], []
                    for j in js2:
                        k = (j["item_id"], "G2", j["temp"], j["rep"]); k0 = (j["item_id"], "G0", j["temp"], j["rep"])
                        if k in resp and k0 in resp:
                            s1, s0 = emb.sim([resp[k], resp[k]], [items[j["item_id"]]["answer_truth"], resp[k0]])
                            d_self.append(1 - s1); d_g0.append(1 - s0)
                    if d_self:
                        pull = round(st.mean(d_g0) / (st.mean(d_g0) + st.mean(d_self)), 3)
                out.append(f"| {name} | {len(S)} | {mean([j['scores']['stance'] for j in js0])} | {mean([j['scores']['stance'] for j in js2])} | {pull} |")
            out.append("\n끌림 지수 = dist(G2–G0)/[dist(G2–G0)+dist(G2–본인)]: 1에 가까우면 본인 쪽, 0.5 미만이면 G0 쪽(이념 편향 의심)")

    # 표7 E1
    if "G3" in conds and "G0" in conds:
        out.append("\n## 표7. E1 누출·순기여 (stance 평균)\n\n| 구분 | G0 | G3 | G2 | leak=G3−G0 | 순기여=G2−G3 |\n|---|---:|---:|---:|---:|---:|")
        for key, vals in [("전체", [None]), ("fame", sorted({str(x.get("fame")) for x in items.values()})), ("strict", ["True", "False"])]:
            for v in vals:
                def m(c):
                    js = [j for j in J if j["cond"] == c and j["temp"] > 0 and j["scores"] and (v is None or str(items[j["item_id"]].get(key)) == v)]
                    return item_mean(js, lambda j: j["scores"]["stance"])
                a0, a3, a2 = m("G0"), m("G3"), m("G2")
                lk = round(a3 - a0, 2) if a0 is not None and a3 is not None else ""
                nc = round(a2 - a3, 2) if a2 is not None and a3 is not None else ""
                out.append(f"| {key}{'='+v if v else ''} | {a0} | {a3} | {a2} | {lk} | {nc} |")

    # 표8 F2
    f2 = [r for r in R if r["format"] == "F2" and r["parsed"] is not None and items[r["item_id"]].get("stance_score_truth")]
    if f2:
        out.append("\n## 표8. F2 7점 동의도 acc = 1−|응답−본인|/6\n\n| 조건 | n | acc |\n|---|---:|---:|")
        for c in conds:
            xs = [1 - abs(r["parsed"] - items[r["item_id"]]["stance_score_truth"]) / 6 for r in f2 if r["cond"] == c and r["temp"] > 0]
            out.append(f"| {c} | {len(xs)} | {mean(xs)} |")

    os.makedirs(P("experiments/results"), exist_ok=True)
    md = P("experiments/results", f"{a.exp}_summary.md")
    open(md, "w", encoding="utf-8").write("\n".join(out) + "\n")
    import csv
    with open(P("experiments/results", f"{a.exp}_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "cond", "n", *KEYS, "sim_emb"]); w.writeheader(); w.writerows(rows)
    print("\n".join(out)); print(f"\n→ {md}")


if __name__ == "__main__":
    main()
