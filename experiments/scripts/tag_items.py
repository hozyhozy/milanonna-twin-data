#!/usr/bin/env python3
"""② 태깅: 판정/태거 모델이 holdout50.json 각 문항의 본인 답에 valence·ground·stance를 1차로 단다.
연구자가 json을 직접 고치고 reviewed=true 로 바꾼다. 모델 원출력은 tagging_log.md에 남긴다.
fame(유명/신규)은 모델이 판단하기 어려우므로 연구자가 직접 채운다.

python experiments/scripts/tag_items.py            # 미태깅 문항만 호출
python experiments/scripts/tag_items.py --force    # 전부 재태깅 (reviewed=true 는 건너뜀)
"""
import json, os, sys, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import LLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITEMS = os.path.join(ROOT, "experiments/items/holdout50.json")
LOG = os.path.join(ROOT, "experiments/items/tagging_log.md")
PROMPT = open(os.path.join(ROOT, "experiments/prompts/tag_items.txt"), encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    obj = json.load(open(ITEMS, encoding="utf-8"))
    llm = LLM(role="tagger")
    n = 0
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n## 태깅 실행 {datetime.date.today()} — {llm.provider}/{llm.model}\n\n")
        for it in obj["items"]:
            if it.get("reviewed"):
                continue
            if it.get("valence_truth") is not None and not a.force:
                continue
            user = PROMPT.replace("{question}", it["question"]).replace("{answer}", it["answer_truth"])
            text, meta = llm.chat(system="당신은 텍스트 코딩을 돕는 연구 보조원입니다. JSON만 출력합니다.",
                                  user=user, temperature=0, json_mode=True, seed=0)
            try:
                j = json.loads(text)
            except Exception:
                log.write(f"- {it['item_id']}: JSON 파싱 실패\n```\n{text}\n```\n"); continue
            it.update({"valence_truth": j.get("valence"), "ground_type_truth": j.get("ground"),
                       "stance_truth": j.get("stance"), "stance_score_truth": j.get("stance_score"),
                       "tag_source": f"{llm.provider}/{llm.model}", "reviewed": False})
            log.write(f"- **{it['item_id']}** [{it['stratum']}] {it['question']}\n"
                      f"  - valence {j.get('valence')} ({j.get('valence_reason')}) · ground {j.get('ground')} ({j.get('ground_reason')})\n"
                      f"  - stance: {j.get('stance')} ({j.get('stance_score')})\n")
            n += 1
    json.dump(obj, open(ITEMS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n}문항 태깅 → {ITEMS}, 로그 {LOG}. 이제 json을 열어 검수하고 reviewed=true, fame 채우세요.")


if __name__ == "__main__":
    main()
