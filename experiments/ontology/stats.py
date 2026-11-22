#!/usr/bin/env python3
"""④ 통계와 역량 질문: 파일럿에서 설계를 고칠 근거가 되는 숫자를 낸다.

python3 experiments/ontology/stats.py --run pilot1
출력: stats.md, tag_growth.csv
"""
import argparse, json, os
from collections import Counter, defaultdict
from common import run_dir, read_jsonl, write_csv, SITUATIONS


def pct(a, b):
    return f"{(100 * a / b):.0f}%" if b else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    a = ap.parse_args()
    d = run_dir(a.run)
    g = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    recs = read_jsonl(os.path.join(d, "extract.jsonl"))
    st = [b for b in g["boxes"] if b["type"] == "Stance"]
    ep = [b for b in g["boxes"] if b["type"] == "Episode"]
    ns = len(g["statements"])
    L = [f"# 파일럿 통계 — {a.run}", "", f"발화 {ns}개 → 입장 상자 {len(st)}개, 일화 {len(ep)}개", ""]

    # 1. 쪼개지나
    per = Counter(len(r["ext"]["stances"]) for r in recs)
    L += ["## 1. 발화가 칸으로 쪼개지나", "",
          f"- 입장이 하나 이상 나온 발화: {pct(sum(v for k, v in per.items() if k > 0), ns)} "
          f"(발화당 입장 수 분포: " + ", ".join(f"{k}개 {v}" for k, v in sorted(per.items())) + ")",
          f"- 입장 상자 중 특징(A) 칸 채움: {pct(sum(1 for b in st if b['hasAttribute']), len(st))}",
          f"- 결과(C) 칸 채움: {pct(sum(1 for b in st if b['hasConsequence']), len(st))}",
          f"- 가치(V) 칸 채움: {pct(sum(1 for b in st if b['hasValue']), len(st))}",
          f"- A·C·V 모두 채움: {pct(sum(1 for b in st if b['hasAttribute'] and b['hasConsequence'] and b['hasValue']), len(st))}",
          f"- A만 비고 C·V가 있음: {pct(sum(1 for b in st if not b['hasAttribute'] and (b['hasConsequence'] or b['hasValue'])), len(st))} "
          "(ACV가 제품용이라 A 칸이 자주 빌 수 있다는 우려 확인용)", ""]

    # 2. 꼬리표 수렴
    seen = {"A": set(), "C": set(), "V": set()}
    growth = []
    for i, r in enumerate(recs, 1):
        for s in r["ext"]["stances"]:
            if s["attribute"]: seen["A"].add(s["attribute"]["tag"])
            if s["consequence"]: seen["C"].add(s["consequence"]["tag"])
            if s["value"]: seen["V"].add(s["value"]["words"])
        growth.append([i, len(seen["A"]), len(seen["C"]), len(seen["V"])])
    write_csv(os.path.join(d, "tag_growth.csv"), ["발화 수", "특징 꼬리표", "결과 꼬리표", "가치(본인 말)"], growth)
    step = max(1, len(growth) // 8)
    L += ["## 2. 꼬리표가 수렴하나 (누적 개수)", "", "| 발화 수 | 특징 | 결과 | 가치 |", "|---|---|---|---|"]
    L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in growth[step - 1::step]]
    for k, name in [("Attribute", "특징"), ("Consequence", "결과"), ("Value", "가치")]:
        pts = g["points"][k]
        once = sum(1 for v in pts.values() if len(v) == 1)
        top = sorted(pts.items(), key=lambda t: -len(t[1]))[:10]
        L += ["", f"- {name}: {len(pts)}개 중 한 번만 쓰인 것 {pct(once, len(pts))}. 많이 쓰인 것: " +
              ", ".join(f"{t}({len(v)})" for t, v in top)]
    L += ["", "곡선이 계속 직선으로 오르면 꼬리표가 너무 잘게 쪼개진 것입니다. 한 번만 쓰인 꼬리표 비율이 높으면 합치기가 필요합니다.", ""]

    # 3. 근거·상황·Schwartz
    ev = Counter(b["evidence"] for b in st)
    sit = Counter(s["situation"] for s in g["statements"].values())
    sch = Counter(x for b in st for x in b["schwartz"])
    eff = Counter(b["valueEffect"] for b in st if b["hasValue"])
    L += ["## 3. 근거 등급 · 질문 상황 · 가치", "",
          "- 근거 등급(찬반): " + ", ".join(f"{k} {pct(v, len(st))}" for k, v in ev.most_common()),
          "- 근거 등급(가치): " + ", ".join(f"{k} {v}" for k, v in Counter(b.get('valueEvidence') for b in st if b['hasValue']).most_common()),
          "- 특질 단서(상위): " + ", ".join(f"{k}({v})" for k, v in Counter(t for s in g['statements'].values() for t in s.get('trait_cues', [])).most_common(10)),
          f"- 교정본으로 추출한 발화: {sum(1 for r in recs if r.get('cleaned'))}/{ns}",
          "- 질문 상황: " + ", ".join(f"{k} {v}" for k, v in sit.most_common()),
          "- Schwartz 칸: " + ", ".join(f"{k} {v}" for k, v in sch.most_common()),
          "- 가치 방향: " + ", ".join(f"{k} {v}" for k, v in eff.items()),
          f"- 구절이 원문과 일치: {pct(sum(1 for b in st for k, ok in b['grounded'].items() if ok), sum(1 for b in st for k, ok in b['grounded'].items() if ok is not None))}", ""]

    # 4. 역량 질문 (자동으로 답해 볼 수 있는 것)
    L += ["## 4. 역량 질문 자동 점검", ""]
    topics = sorted(g["points"]["Topic"].items(), key=lambda t: -len(t[1]))
    if topics:
        t, bids = topics[0]
        bs = [b for b in st if b["id"] in bids]
        L.append(f"- CQ01 (P01은 '{t}'에 찬성/반대?): " + ", ".join(f"{b['hasPolarity']}({b['id']})" for b in bs[:5]))
        b = next((b for b in bs if b["hasAttribute"] or b["hasValue"]), bs[0])
        L.append(f"- CQ02 (왜?): {b['hasTopic']} → 특징 {b['hasAttribute']} → 결과 {b['hasConsequence']} → 가치 {b['hasValue']} {b['schwartz']}")
        L.append(f"- CQ03 (근거 발화): {b['evidencedBy'][0]} — \"{(b['phrases']['topic'] or '')[:60]}\"")
    L.append(f"- CQ04 (가장 자주 지키려는 가치): " + ", ".join(f"{k} {v}" for k, v in sch.most_common(3)))
    by = defaultdict(set)
    for b in st:
        by[b["hasTopic"]].add(b["hasPolarity"])
    L.append(f"- CQ07 (입장이 바뀐 후보 = 같은 대상에 찬반이 다른 경우): {sum(1 for v in by.values() if len(v) > 1)}건")
    L.append(f"- CQ14 (근거가 추론뿐인 상자): {ev.get('추론', 0)}개")
    L.append("- CQ05·CQ06·CQ12 (다른 사람과의 비교): 한 사람 자료라 답할 수 없음 → 다른 인물 자료가 필요")
    open(os.path.join(d, "stats.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
