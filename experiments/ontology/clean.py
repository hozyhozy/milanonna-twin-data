#!/usr/bin/env python3
"""⓪ 자막 교정 (선택): 자동 자막의 인식 오류만 최소로 고친다. 원본(answer_raw)은 그대로 두고 따로 저장한다.

원칙
  - 잘못 인식된 단어만 고친다. 말을 다듬거나 요약하거나 덧붙이지 않는다.
  - "~거든요", "그러니까", "아이고" 같은 말버릇과 어순은 그대로 둔다 (말투 분석에 필요).
  - 알아들을 수 없는 부분은 고치지 말고 [?]로 표시한다.
  - 안전장치: 원문과 교정본의 글자 유사도가 --min-sim 보다 낮으면 '고쳐 쓴 것'으로 보고 거절(원문 사용).

python3 experiments/ontology/clean.py --split Train            # Train 381개
python3 experiments/ontology/clean.py --split Holdout --ids-from-holdout50
출력: experiments/runs/ontology/_clean/<split>.jsonl   (모든 run이 같이 씀)
"""
import argparse, difflib, json, os
from common import load_split, read_jsonl, parse_json, CLEAN_DIR, ROOT

SYSTEM = """너는 한국어 자동 자막을 교정하는 편집자다. 잘못 인식된 단어만 문맥에 맞게 고친다.
말을 다듬거나 요약하거나 문장을 새로 쓰지 않는다. 말버릇·반복·어순·구어체 어미는 그대로 둔다.
확신이 없는 부분은 고치지 말고 [?]를 붙인다. 반드시 JSON으로만 답한다."""

USER = """[질문]
{q}

[자동 자막 답]
{a}

JSON 형식: {{"clean": "교정한 답 전체", "changes": [{{"from": "원래 말", "to": "고친 말"}}], "unsure": ["확신 없는 부분"]}}"""


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Train")
    ap.add_argument("--ids-from-holdout50", action="store_true", help="Holdout은 holdout50 문항만")
    ap.add_argument("--min-sim", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out_p = os.path.join(CLEAN_DIR, f"{a.split}.jsonl")
    done = {r["id"] for r in read_jsonl(out_p)}
    items = load_split(a.split)
    if a.ids_from_holdout50:
        keep = {x["item_id"] for x in json.load(open(os.path.join(ROOT, "experiments/items/holdout50.json"), encoding="utf-8"))["items"]}
        items = [x for x in items if x["id"] in keep]
    if a.limit:
        items = items[:a.limit]
    mock = os.environ.get("MOCK_LLM") == "1"
    if not mock:
        from llm import LLM
        llm = LLM(role="tagger")
    n_ok = n_rej = 0
    with open(out_p, "a", encoding="utf-8") as f:
        for i, x in enumerate(items):
            if x["id"] in done:
                continue
            raw = x.get("answer_raw") or x["answer"]
            if mock:
                res = {"clean": raw.replace("이쁘지 프", "이쁜"), "changes": [], "unsure": []}
            else:
                text, _ = llm.chat(system=SYSTEM, user=USER.format(q=x["question"], a=raw), temperature=0,
                                   max_tokens=2000, json_mode=True)
                res = parse_json(text) or {}
            clean = (res.get("clean") or "").strip()
            s = sim(raw, clean) if clean else 0.0
            ok = bool(clean) and s >= a.min_sim
            n_ok += ok; n_rej += (not ok)
            f.write(json.dumps({"id": x["id"], "split": a.split, "raw": raw, "clean": clean, "sim": round(s, 3),
                                "accepted": ok, "changes": res.get("changes", []), "unsure": res.get("unsure", [])},
                               ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i+1}/{len(items)}] {x['id']} 유사도 {s:.2f} {'채택' if ok else '거절(원문 사용)'}")
    print(f"완료: 채택 {n_ok}, 거절 {n_rej} → {out_p}")


if __name__ == "__main__":
    main()
