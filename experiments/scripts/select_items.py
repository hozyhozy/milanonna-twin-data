#!/usr/bin/env python3
"""② Holdout 50문항 층화 추출 → experiments/items/holdout50.json (가명본 기준).

층(설계서 §2.2): 의견·가치관형 25 / 일화 회상형 10 / 채널·팬·경력 10 / 기술·SNS·AI 5
규칙은 topic 키워드 + 정규식 휴리스틱. 결과는 연구자가 훑어 교체 가능(--swap 옵션 없이 json 직접 편집).
"""
import json, os, re, random, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALUE_TOPICS = ["가치관", "인생관", "세대", "연애", "결혼", "육아", "나이듦", "소비", "조언", "인간관계", "관계", "행복",
                "후회", "의사결정", "나눔", "예의", "자기인식", "노후", "사회", "신앙", "언어습관", "태도"]
CHANNEL_TOPICS = ["동영상 채널", "팬", "경력", "방송", "직장", "유학"]
TECH_RE = re.compile(r"AI|인공지능|알고리즘|SNS|인스타|댓글|스마트폰|온라인|디지털|기술")
ANEC_RE = re.compile(r"(\d{2,4}년|\d{1,2}살|그때|그 당시|어렸을 때|젊었을 때|옛날에|처음에|기억에 남)")
QUOTA = {"value": 25, "anecdote": 10, "channel": 10, "tech": 5}


def stratum(x):
    text = x["question"] + " " + x["answer"]
    if TECH_RE.search(x["question"]) or (TECH_RE.search(text) and "동영상 채널" not in x["topic"]):
        return "tech"
    if ANEC_RE.search(x["answer"]) and x["type"] == "interview" or ANEC_RE.search(x["question"]):
        return "anecdote"
    if any(t in x["topic"] for t in CHANNEL_TOPICS):
        return "channel"
    if any(t in x["topic"] for t in VALUE_TOPICS):
        return "value"
    return "anecdote" if ANEC_RE.search(x["answer"]) else "value"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--src", default=os.path.join(ROOT, "data/anon/Holdout.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments/items/holdout50.json"))
    a = ap.parse_args()
    items = json.load(open(a.src, encoding="utf-8"))["items"]
    items = [x for x in items if x["split"] == "Holdout" and x.get("overlap") in (None, "novel")]
    for x in items:
        x["stratum"] = stratum(x)
    pool = collections.defaultdict(list)
    for x in items:
        pool[x["stratum"]].append(x)
    print("후보 수:", {k: len(v) for k, v in pool.items()})
    rng = random.Random(a.seed)
    chosen = []
    for s, n in QUOTA.items():
        cand = pool[s][:]
        rng.shuffle(cand)
        # 같은 영상에서 너무 많이 뽑히지 않도록 영상당 최대 3개
        per_video = collections.Counter()
        for x in cand:
            if len([c for c in chosen if c["stratum"] == s]) >= n:
                break
            if per_video[x["video_id"]] >= 3:
                continue
            per_video[x["video_id"]] += 1
            chosen.append(x)
    short = {k: sum(1 for c in chosen if c["stratum"] == k) for k in QUOTA}
    print("선정:", short, "(부족한 층은 그대로 보고)")
    out = []
    for x in chosen:
        out.append({"item_id": x["id"], "video_id": x["video_id"], "stratum": x["stratum"], "type": x["type"],
                    "topic": x["topic"], "year": x["year"], "strict": x["split_strict"] == "Holdout",
                    "question": x["question"], "answer_truth": x["answer"],
                    "valence_truth": None, "ground_type_truth": None, "fame": None,
                    "stance_truth": None, "stance_score_truth": None, "tag_source": None, "reviewed": False})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"seed": a.seed, "quota": QUOTA, "n": len(out), "items": out}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {a.out} ({len(out)}문항)")


if __name__ == "__main__":
    main()
