#!/usr/bin/env python3
"""③ 공리 검사: graph.json 을 설계표 v0.4 공리로 검사한다 (닫힌 세계: 빠진 것도 위반).

python3 experiments/ontology/check.py --run pilot1
출력: violations.csv, check.md
"""
import argparse, json, os
from collections import Counter, defaultdict
from common import run_dir, write_csv, SITUATIONS, UNKNOWN_SITUATION, POLARITY, EVIDENCE, EFFECT, SCHWARTZ_OK

AXIOMS = {
    "A02": "Stance는 hasHolder·hasTopic·hasPolarity를 정확히 하나씩 가진다",
    "A03": "hasValue가 있으면 valueEffect(지킴/해침)가 정확히 하나 있다",
    "A04": "Value의 Schwartz 칸은 10칸 중 1~2개 또는 '해당 없음'",
    "A05": "모든 상자는 evidencedBy로 Statement 하나 이상에 닿는다",
    "A06": "TraitClaim은 상황 2종 이상 — 이번 파일럿은 특질을 추출하지 않음(해당 없음)",
    "A07": "Episode는 본인 Statement에만 근거한다",
    "A09": "Situation은 정해진 목록 안 (만든 질문은 '알 수 없음')",
    "A10": "(보고) 가치 근거가 '추론'인 상자 — 비슷한 사람 찾기에서 제외될 상자 수",
    "A11": "뒤 시기(Holdout·Retest) Statement는 관계도에 들어갈 수 없다",
    "A12": "같은 사람·같은 대상에 찬반이 다른 Stance가 있으면 revises로 이어져야 한다 (후보로 보고)",
    "A13": "특징·결과의 아래층 클래스(세부 꼬리표)는 2명 이상 공유해야 한다 (1명 자료에서는 모두 '대기')",
    "A19": "결과(C) 꼬리표는 Schwartz 가치의 이름을 그대로 쓰지 않는다 (그러면 hasValue로 옮김)",
    "A16": "상자의 근거 발화는 모두 그 hasHolder 본인의 것이다",
    "A17": "상자 ID는 겹치지 않는다",
    "A18": "근거 발화 ID의 사람 번호 = 상자의 hasHolder",
    "G1": "(보조) 구절이 답 원문에 그대로 있다 — 모델이 구절을 지어냈는지 확인",
    "E1": "(보조) 근거 등급은 직접 진술/관찰/추론 중 하나",
}


def check(g):
    V = []
    st = g["statements"]
    ids = Counter(b["id"] for b in g["boxes"])
    for i, c in ids.items():
        if c > 1:
            V.append(("A17", i, f"ID {c}번 사용"))
    for sid, s in st.items():
        if s["situation"] not in SITUATIONS + [UNKNOWN_SITUATION]:
            V.append(("A09", sid, s["situation"]))
        if s["split"] != "Train":
            V.append(("A11", sid, f"{s['split']} 발화가 관계도에 있음"))
    for b in g["boxes"]:
        bid = b["id"]
        ev = b.get("evidencedBy") or []
        if not ev or any(e not in st for e in ev):
            V.append(("A05", bid, "근거 발화 없음"))
        for e in ev:
            if e.split("-")[0] != b.get("hasHolder"):
                V.append(("A18", bid, f"{e} ≠ {b.get('hasHolder')}"))
                V.append(("A16", bid, "남의 발화가 근거"))
        if b["type"] == "Episode":
            if not b.get("grounded"):
                V.append(("G1", bid, "일화 구절이 원문에 없음"))
            continue
        for f in ["hasHolder", "hasTopic", "hasPolarity"]:
            if not b.get(f):
                V.append(("A02", bid, f"{f} 없음"))
        if b.get("hasPolarity") and b["hasPolarity"] not in POLARITY:
            V.append(("A02", bid, f"찬반 값 '{b['hasPolarity']}'"))
        if b.get("hasValue"):
            if b.get("valueEffect") not in EFFECT:
                V.append(("A03", bid, f"valueEffect '{b.get('valueEffect')}'"))
            sch = b.get("schwartz") or []
            if not (1 <= len(sch) <= 2) or any(x not in SCHWARTZ_OK for x in sch):
                V.append(("A04", bid, f"Schwartz {sch}"))
        if b.get("evidence") not in EVIDENCE:
            V.append(("E1", bid, str(b.get("evidence"))))
        if b.get("hasValue") and b.get("valueEvidence") not in EVIDENCE:
            V.append(("E1", bid, f"가치 근거 {b.get('valueEvidence')}"))
        if b.get("hasValue") and b.get("valueEvidence") == "추론":
            V.append(("A10", bid, "가치 근거가 추론"))
        for k, ok in (b.get("grounded") or {}).items():
            if ok is False:
                V.append(("G1", bid, f"{k} 구절이 원문에 없음"))
    # A12: 같은 사람·대상, 다른 찬반
    by = defaultdict(list)
    for b in g["boxes"]:
        if b["type"] == "Stance":
            by[(b["hasHolder"], b["hasTopic"])].append(b)
    for (h, t), bs in by.items():
        if len({b["hasPolarity"] for b in bs}) > 1:
            V.append(("A12", ",".join(b["id"] for b in bs), f"{h} · {t}: " + " / ".join(f"{b['id']}={b['hasPolarity']}" for b in bs)))
    # A13
    for k in ["Attribute", "Consequence"]:   # v0.4: 특징·결과만. 가치는 Schwartz 칸이 처음부터 클래스
        for tag, bids in g["points"][k].items():
            holders = {next(b for b in g["boxes"] if b["id"] == i)["hasHolder"] for i in bids}
            if len(holders) < 2:
                V.append(("A13", f"{k}:{tag}", f"{len(holders)}명 → 대기"))
    # A19: 결과 꼬리표에 Schwartz 가치 이름이 그대로 들어가면 V로 옮길 후보 (뜻만 같은 경우는 사람이 확인)
    from common import SCHWARTZ
    for tag in g["points"]["Consequence"]:
        hit = [v for v in SCHWARTZ if v in tag]
        if hit:
            V.append(("A19", f"Consequence:{tag}", f"가치 이름 {', '.join(hit)} → hasValue로 옮길 것"))
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    a = ap.parse_args()
    d = run_dir(a.run)
    g = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    V = check(g)
    write_csv(os.path.join(d, "violations.csv"), ["공리", "대상", "내용"], V)
    c = Counter(v[0] for v in V)
    n_box = len(g["boxes"])
    lines = ["# 공리 검사 결과", "", f"상자 {n_box}개, 발화 {len(g['statements'])}개", "",
             "| 공리 | 내용 | 위반 수 |", "|---|---|---|"]
    for k, t in AXIOMS.items():
        lines.append(f"| {k} | {t} | {c.get(k, 0)} |")
    lines += ["", "A13은 한 사람 자료에서는 모든 꼬리표가 '대기'로 나오는 것이 정상입니다. 다른 사람 자료가 들어와야 클래스가 생깁니다.",
              "A12는 오류가 아니라 '입장이 바뀌었거나 상황에 따라 달라지는' 후보입니다. 사람이 보고 revises로 이을지 정합니다.",
              "G1이 많으면 모델이 구절을 요약·변형한 것이니 프롬프트를 고칩니다."]
    open(os.path.join(d, "check.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[4:4 + len(AXIOMS) + 2]))


if __name__ == "__main__":
    main()
