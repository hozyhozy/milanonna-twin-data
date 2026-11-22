#!/usr/bin/env python3
"""② 상자 만들기: 추출 결과를 입장 상자(ST-)·일화 상자(EP-)와 공유 점(Topic·Attribute·Consequence·Value)으로 바꾼다.

python3 experiments/ontology/build.py --run pilot1

출력 (runs/ontology/<run>/):
  graph.json        상자·점·발화 (공리 검사와 답변 생성이 읽는 관계도)
  sheet_발화.csv     코딩 시트 '발화' 탭과 같은 열 순서
  sheet_코딩.csv     코딩 시트 '코딩' 탭과 같은 열 순서 (코더 = 모델)
  sheet_꼬리표.csv   코딩 시트 '꼬리표' 탭과 같은 열 순서
"""
import argparse, json, os
from collections import defaultdict
from common import run_dir, read_jsonl, write_csv, PERSON


def build(recs):
    boxes, statements = [], {}
    points = {k: defaultdict(list) for k in ["Topic", "Attribute", "Consequence", "Value"]}
    n_st = n_ep = 0
    for r in recs:
        sid, ext = r["statement_id"], r["ext"]
        holder = sid.split("-")[0]
        statements[sid] = {"item_id": r["item_id"], "split": r["split"], "year": r.get("year"),
                           "question": r["question"], "answer": r["answer"], "situation": ext["situation"],
                           "style": ext.get("style", ""), "video_id": r["video_id"], "type": r.get("type"),
                           "trait_cues": ext.get("trait_cues", [])}
        for s in ext["stances"]:
            n_st += 1
            bid = f"ST-{n_st:04d}"
            b = {"id": bid, "type": "Stance", "hasHolder": holder, "hasTopic": s["topic"],
                 "hasPolarity": s["polarity"], "label": f"{holder} · {s['topic']} · {s['polarity']}",
                 "hasAttribute": s["attribute"]["tag"] if s["attribute"] else None,
                 "hasConsequence": s["consequence"]["tag"] if s["consequence"] else None,
                 "hasValue": s["value"]["words"] if s["value"] else None,
                 "schwartz": s["value"]["schwartz"] if s["value"] else [],
                 "valueEffect": s["value"]["effect"] if s["value"] else None,
                 "inSituation": ext["situation"], "evidencedBy": [sid], "evidence": s["evidence"],
                 "valueEvidence": (s["value"] or {}).get("evidence"),
                 "phrases": {"topic": s["topic_phrase"],
                             "attribute": s["attribute"]["phrase"] if s["attribute"] else None,
                             "consequence": s["consequence"]["phrase"] if s["consequence"] else None,
                             "value": s["value"]["phrase"] if s["value"] else None},
                 "grounded": {"topic": s["topic_grounded"],
                              "attribute": s["attribute"]["grounded"] if s["attribute"] else None,
                              "consequence": s["consequence"]["grounded"] if s["consequence"] else None,
                              "value": s["value"]["grounded"] if s["value"] else None},
                 "new": {"attribute": (s["attribute"] or {}).get("new"), "consequence": (s["consequence"] or {}).get("new"),
                         "value": (s["value"] or {}).get("new")}}
            boxes.append(b)
            points["Topic"][b["hasTopic"]].append(bid)
            for k, f in [("Attribute", "hasAttribute"), ("Consequence", "hasConsequence"), ("Value", "hasValue")]:
                if b[f]:
                    points[k][b[f]].append(bid)
        for e in ext["episodes"]:
            n_ep += 1
            boxes.append({"id": f"EP-{n_ep:04d}", "type": "Episode", "hasHolder": holder,
                          "label": f"{holder} · {e['label']}", "evidencedBy": [sid], "phrase": e["phrase"],
                          "grounded": e["grounded"]})
    return {"persons": sorted({s.split('-')[0] for s in statements}), "statements": statements,
            "boxes": boxes, "points": {k: dict(v) for k, v in points.items()}}


def export_sheets(g, recs, d):
    write_csv(os.path.join(d, "sheet_발화.csv"),
              ["발화 ID", "질문 (요약)", "답 원문", "질문 상황", "코딩 상태", "말투 메모", "주변 맥락"],
              [[sid, s["question"], s["answer"], s["situation"], "코딩 중", s["style"], f"{s['year']} · {s['video_id']}"]
               for sid, s in g["statements"].items()])
    rows = []
    for b in g["boxes"]:
        sid = b["evidencedBy"][0]
        if b["type"] == "Episode":
            rows.append([sid, b["id"], b["phrase"], "일화", b["label"].split(" · ", 1)[1], "", "", "직접 진술", "모델", ""])
            continue
        ev = b["evidence"]
        rows.append([sid, b["id"], b["phrases"]["topic"], "입장-대상", b["hasTopic"], "", "", ev, "모델", ""])
        rows.append([sid, b["id"], b["phrases"]["topic"], "입장-찬반", b["hasPolarity"], "", "", ev, "모델", ""])
        if b["hasAttribute"]:
            rows.append([sid, b["id"], b["phrases"]["attribute"], "특징 A", b["hasAttribute"], "", "", ev, "모델",
                         "" if b["grounded"]["attribute"] else "구절이 원문과 다름"])
        if b["hasConsequence"]:
            rows.append([sid, b["id"], b["phrases"]["consequence"], "결과 C", b["hasConsequence"], "", "", ev, "모델",
                         "" if b["grounded"]["consequence"] else "구절이 원문과 다름"])
        if b["hasValue"]:
            rows.append([sid, b["id"], b["phrases"]["value"], f"가치 V ({b['valueEffect']})", b["hasValue"], "",
                         ", ".join(b["schwartz"]), b.get("valueEvidence") or ev, "모델", "" if b["grounded"]["value"] else "구절이 원문과 다름"])
    write_csv(os.path.join(d, "sheet_코딩.csv"),
              ["발화 ID", "상자 ID", "구절 (원문 그대로)", "칸", "세부 꼬리표", "새로/재사용", "Schwartz 칸", "근거 등급", "코더", "메모"], rows)
    tag_rows = []
    for k, slot in [("Attribute", "특징 A"), ("Consequence", "결과 C"), ("Value", "가치 V")]:
        f = {"Attribute": "attribute", "Consequence": "consequence", "Value": "value"}[k]
        for tag, bids in sorted(g["points"][k].items(), key=lambda t: -len(t[1])):
            first = next(b for b in g["boxes"] if b["id"] == bids[0])
            tag_rows.append([slot, tag, "", "", first["phrases"][f] or "", len(bids),
                             len({next(b for b in g['boxes'] if b['id'] == i)['hasHolder'] for i in bids}), "가안", first["evidencedBy"][0]])
    write_csv(os.path.join(d, "sheet_꼬리표.csv"),
              ["칸", "세부 꼬리표", "정의", "판별 기준", "예 구절", "사용 횟수", "사람 수", "상태", "처음 나온 발화"], tag_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    a = ap.parse_args()
    d = run_dir(a.run)
    recs = read_jsonl(os.path.join(d, "extract.jsonl"))
    if not recs:
        raise SystemExit("extract.jsonl 이 없습니다. extract.py 를 먼저 실행하세요.")
    g = build(recs)
    json.dump(g, open(os.path.join(d, "graph.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    export_sheets(g, recs, d)
    n_st = sum(b["type"] == "Stance" for b in g["boxes"])
    print(f"발화 {len(g['statements'])} → 입장 상자 {n_st}, 일화 {len(g['boxes']) - n_st}, "
          f"점: 대상 {len(g['points']['Topic'])} · 특징 {len(g['points']['Attribute'])} · "
          f"결과 {len(g['points']['Consequence'])} · 가치 {len(g['points']['Value'])}")


if __name__ == "__main__":
    main()
