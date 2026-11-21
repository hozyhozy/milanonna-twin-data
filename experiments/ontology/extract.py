#!/usr/bin/env python3
"""① 추출: 발화(질문+답)를 입장 상자 재료로 분해한다. 꼬리표 목록은 읽어 나가며 자란다.

python3 experiments/ontology/extract.py --run pilot1 --limit 30          # Train 앞 30발화
python3 experiments/ontology/extract.py --run pilot1                     # Train 전체(381), 이어서 실행 가능
python3 experiments/ontology/extract.py --run pilot2 --tags 꼬리표.csv    # 검토한 꼬리표 목록에서 시작
MOCK_LLM=1 ... 로 API 없이 형식 점검.

출력: runs/ontology/<run>/extract.jsonl, tags.json
"""
import argparse, json, os
from common import (load_split, video_index, statement_id, run_dir, read_jsonl, TagBook, EXTRACT_SYSTEM,
                    extract_prompt, parse_json, mock_extract, SITUATIONS, POLARITY, EVIDENCE, EFFECT, SCHWARTZ_OK)


def clean(ext, answer):
    """형식을 정리하고, 구절이 원문에 실제로 있는지(grounded) 표시한다."""
    ext = ext or {}
    def g(p):
        return bool(p) and p.strip() in answer
    out = {"situation": ext.get("situation") if ext.get("situation") in SITUATIONS else "기타",
           "situation_raw": ext.get("situation"), "stances": [], "episodes": [], "style": ext.get("style") or ""}
    for s in ext.get("stances") or []:
        if not isinstance(s, dict) or not s.get("topic"):
            continue
        st = {"topic": str(s.get("topic")).strip(), "polarity": s.get("polarity"),
              "evidence": s.get("evidence") if s.get("evidence") in EVIDENCE else "추론",
              "topic_phrase": s.get("topic_phrase") or "", "topic_grounded": g(s.get("topic_phrase"))}
        for k in ["attribute", "consequence"]:
            v = s.get(k)
            st[k] = ({"tag": str(v.get("tag")).strip(), "phrase": v.get("phrase") or "", "grounded": g(v.get("phrase"))}
                     if isinstance(v, dict) and v.get("tag") else None)
        v = s.get("value")
        if isinstance(v, dict) and v.get("words"):
            sch = v.get("schwartz") or []
            sch = [sch] if isinstance(sch, str) else list(sch)
            st["value"] = {"words": str(v.get("words")).strip(), "schwartz": sch, "effect": v.get("effect"),
                           "phrase": v.get("phrase") or "", "grounded": g(v.get("phrase"))}
        else:
            st["value"] = None
        out["stances"].append(st)
    for e in ext.get("episodes") or []:
        if isinstance(e, dict) and e.get("label"):
            out["episodes"].append({"label": e["label"], "phrase": e.get("phrase") or "", "grounded": g(e.get("phrase"))})
    return out


def extract_one(llm, question, answer, tagbook, seed=0):
    if llm is None:
        raw, meta = mock_extract(question, answer, seed), {"model": "mock"}
    else:
        text, meta = llm.chat(system=EXTRACT_SYSTEM, user=extract_prompt(question, answer, tagbook),
                              temperature=0, max_tokens=1500, seed=seed, json_mode=True)
        raw = parse_json(text)
    return clean(raw, answer), meta


def grow(tagbook, ext):
    """새 꼬리표 여부를 표시하고 목록에 넣는다."""
    for s in ext["stances"]:
        for key, slot in [("A", "attribute"), ("C", "consequence")]:
            if s[slot]:
                s[slot]["new"] = tagbook.is_new(key, s[slot]["tag"])
                tagbook.add(key, s[slot]["tag"])
        if s["value"]:
            s["value"]["new"] = tagbook.is_new("V", s["value"]["words"])
            tagbook.add("V", s["value"]["words"])


def get_llm():
    if os.environ.get("MOCK_LLM") == "1":
        return None
    from llm import LLM
    return LLM(role="tagger")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--split", default="Train")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N발화만 (0=전체)")
    ap.add_argument("--tags", default="", help="시작 꼬리표 목록 CSV (코딩 시트 '꼬리표' 탭)")
    a = ap.parse_args()

    d = run_dir(a.run)
    out_p = os.path.join(d, "extract.jsonl")
    done = {r["statement_id"]: r for r in read_jsonl(out_p)}
    vidx = video_index()
    items = load_split(a.split)
    items.sort(key=lambda x: (vidx[x["video_id"]], x["id"]))   # 자료 순 → 발화 순
    if a.limit:
        items = items[:a.limit]

    tb = TagBook.from_csv(a.tags)
    for r in done.values():          # 이어서 실행: 이미 뽑은 꼬리표를 목록에 복원
        grow(tb, r["ext"])
    llm = get_llm()

    n_new = 0
    with open(out_p, "a", encoding="utf-8") as f:
        for i, x in enumerate(items):
            sid = statement_id(x, vidx)
            if sid in done:
                continue
            ext, meta = extract_one(llm, x["question"], x["answer"], tb, seed=0)
            grow(tb, ext)
            rec = {"statement_id": sid, "item_id": x["id"], "video_id": x["video_id"], "split": a.split,
                   "year": x.get("year"), "topic_label": x.get("topic"), "type": x.get("type"),
                   "question": x["question"], "answer": x["answer"], "ext": ext, "model": meta.get("model")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n_new += 1
            print(f"[{i+1}/{len(items)}] {sid} 입장 {len(ext['stances'])}개 · 일화 {len(ext['episodes'])}개 · {ext['situation']}")
    json.dump(tb.dump(), open(os.path.join(d, "tags.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"완료: 새로 추출 {n_new}건 → {out_p}")


if __name__ == "__main__":
    main()
