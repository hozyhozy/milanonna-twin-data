#!/usr/bin/env python3
"""꼬리표 합치기 (코드북 만들기): 자유롭게 붙은 세부 꼬리표를 뜻이 같은 것끼리 묶어 대표 꼬리표를 제안한다.

근거: 래더링 분석의 요약 코드 단계(Reynolds & Gutman 1988), 개방 코딩 → 축 코딩,
      LLM과 사람이 함께 코드북을 만드는 방식(Dai et al. 2023, LLM-in-the-loop).

1) 제안   python3 experiments/ontology/merge.py --run pilot2
          → runs/ontology/pilot2/merge_proposal.csv  (칸, 세부 꼬리표=대표, 정의, 판별 기준, 묶인 꼬리표, 사용 횟수, 상태)
2) 사람이 CSV를 열어 고침: 묶음을 쪼개거나 합치고, 대표 이름을 바꾸고, 상태를 '확정'으로.
3) 적용   python3 experiments/ontology/merge.py --run pilot2 --apply runs/ontology/pilot2/merge_proposal.csv --to pilot2m
          → 새 run(pilot2m)에 꼬리표를 바꾼 extract.jsonl 을 만든다. 그다음 build/check/stats 를 pilot2m 으로.
4) 다음 추출부터는 그 CSV를 --tags 로 넣으면 코드북에서 고른다.
"""
import argparse, csv, json, os
from collections import defaultdict
from common import run_dir, read_jsonl, write_csv, parse_json

SLOT = {"A": ("특징 A", "attribute", "tag"), "C": ("결과 C", "consequence", "tag"), "V": ("가치 V", "value", "words")}

SYSTEM = """너는 질적 연구의 코드북을 정리하는 연구자다. 뜻이 같은 꼬리표끼리 묶어 대표 꼬리표를 정한다.
억지로 묶지 않는다: 뜻이 다르면 따로 둔다. 반드시 JSON으로만 답한다."""

USER = """칸: {slot} — {meaning}

아래는 인터뷰 발화에서 뽑은 세부 꼬리표들이다 (사용 횟수 · 예 구절).
{items}

뜻이 같은 것끼리 묶어라. 모든 꼬리표는 정확히 한 묶음에 들어가야 한다.
대표 꼬리표는 10자 안팎의 일반적인 말로, 여러 사람에게 쓸 수 있게 짓는다.
JSON 형식: {{"groups": [{{"canonical": "대표 꼬리표", "definition": "한 줄 정의", "criterion": "이런 말이 나오면", "members": ["원래 꼬리표", ...]}}]}}"""

MEANING = {"A": "대상이나 선택을 어떤 것이라고 보는가 (성질·규정)",
           "C": "그 때문에 본인에게 생기는 일 (Schwartz 가치의 이름이나 뜻을 그대로 말하는 꼬리표는 대표로 삼지 말고 정의에 '→ V로 옮길 것'이라고 적는다)",
           "V": "결국 왜 중요한가 (지키려는 것)"}


def collect(recs):
    out = {k: defaultdict(lambda: {"n": 0, "ex": []}) for k in SLOT}
    for r in recs:
        for s in r["ext"]["stances"]:
            for k, (_, f, key) in SLOT.items():
                v = s.get(f)
                if v and v.get(key):
                    d = out[k][v[key]]
                    d["n"] += 1
                    if len(d["ex"]) < 2:
                        d["ex"].append(f"{s['topic']}: {v.get('phrase', '')[:40]}")
    return out


def propose(a, d, recs):
    mock = os.environ.get("MOCK_LLM") == "1"
    if not mock:
        from llm import LLM
        llm = LLM(role="tagger")
    rows = []
    for k, tags in collect(recs).items():
        if not tags:
            continue
        names = list(tags)
        if mock:
            groups = [{"canonical": t, "definition": "", "criterion": "", "members": [t]} for t in names]
        else:
            items = "\n".join(f"- {t} ({v['n']}) · " + " / ".join(v["ex"]) for t, v in sorted(tags.items(), key=lambda x: -x[1]["n"]))
            text, _ = llm.chat(system=SYSTEM, user=USER.format(slot=SLOT[k][0], meaning=MEANING[k], items=items),
                               temperature=0, max_tokens=4000, json_mode=True)
            groups = (parse_json(text) or {}).get("groups", [])
        seen = set()
        for g in groups:
            mem = [m for m in g.get("members", []) if m in tags and m not in seen]
            if not mem:
                continue
            seen.update(mem)
            rows.append([SLOT[k][0], g.get("canonical") or mem[0], g.get("definition", ""), g.get("criterion", ""),
                         " | ".join(mem), sum(tags[m]["n"] for m in mem), "가안"])
        for t in names:                      # 모델이 빠뜨린 꼬리표는 혼자인 묶음으로
            if t not in seen:
                rows.append([SLOT[k][0], t, "", "", t, tags[t]["n"], "가안"])
        print(f"{SLOT[k][0]}: 꼬리표 {len(names)}개 → 대표 {sum(1 for r in rows if r[0] == SLOT[k][0])}개")
    p = os.path.join(d, "merge_proposal.csv")
    write_csv(p, ["칸", "세부 꼬리표", "정의", "판별 기준", "묶인 꼬리표", "사용 횟수", "상태"], rows)
    print(f"제안 → {p}  (열어서 고친 뒤 --apply 로 적용)")


def apply(a, d, recs):
    m = {k: {} for k in SLOT}
    by_name = {v[0]: k for k, v in SLOT.items()}
    for r in csv.DictReader(open(a.apply, encoding="utf-8-sig")):
        if (r.get("상태") or "").strip() == "삭제":
            continue
        k = by_name.get((r.get("칸") or "").strip())
        if not k:
            continue
        for mem in (r.get("묶인 꼬리표") or "").split("|"):
            if mem.strip():
                m[k][mem.strip()] = r["세부 꼬리표"].strip()
    out_d = run_dir(a.to)
    n_changed = 0
    with open(os.path.join(out_d, "extract.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            for s in r["ext"]["stances"]:
                for k, (_, fld, key) in SLOT.items():
                    v = s.get(fld)
                    if v and v.get(key) in m[k] and m[k][v[key]] != v[key]:
                        v["orig_" + key] = v[key]; v[key] = m[k][v[key]]; n_changed += 1
            r["merged_from"] = a.run
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tags = {k: {} for k in SLOT}
    for r in recs:
        for s in r["ext"]["stances"]:
            for k, (_, fld, key) in SLOT.items():
                v = s.get(fld)
                if v and v.get(key):
                    tags[k][v[key]] = tags[k].get(v[key], 0) + 1
    json.dump(tags, open(os.path.join(out_d, "tags.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"꼬리표 {n_changed}곳을 대표 꼬리표로 바꿈 → {out_d}/extract.jsonl")
    print(f"다음: build.py / check.py / stats.py 를 --run {a.to} 로 실행")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot2")
    ap.add_argument("--apply", default="", help="사람이 고친 merge_proposal.csv")
    ap.add_argument("--to", default="", help="적용 결과를 넣을 새 run 이름")
    a = ap.parse_args()
    d = run_dir(a.run)
    recs = read_jsonl(os.path.join(d, "extract.jsonl"))
    if not recs:
        raise SystemExit("extract.jsonl 이 없습니다.")
    if a.apply:
        if not a.to:
            raise SystemExit("--to 새 run 이름을 주세요 (원래 run은 그대로 둡니다)")
        apply(a, d, recs)
    else:
        propose(a, d, recs)


if __name__ == "__main__":
    main()
