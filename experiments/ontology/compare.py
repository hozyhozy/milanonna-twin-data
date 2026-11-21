#!/usr/bin/env python3
"""⑥ 같은 추출기로 비교: 실제 답과 트윈 답을 똑같은 추출기로 분해해 칸별로 맞대어 본다.
Etudo & Yoon에서 조직 문서와 새 글을 같은 추출기로 뽑아 관계별로 비교한 것(FRP)에 해당.

python3 experiments/ontology/compare.py --run pilot1
출력: compare.jsonl, compare.md (조건별 충실도 프로필)

항목별 점수
  polarity  실제 답과 트윈 답의 첫 입장 찬반이 같은가 (둘 다 입장이 있을 때만)
  attr      특징 꼬리표 겹침 (자카드)
  value     Schwartz 칸 겹침 (자카드, 둘 중 하나라도 가치가 있을 때)
  episode   트윈이 든 일화 수 (실제 답보다 많으면 지어냈을 가능성)
  char_sim  문자 n-gram 코사인 (참고용 표면 유사도)
"""
import argparse, json, os
from collections import defaultdict
from common import run_dir, read_jsonl, TagBook
from extract import extract_one, get_llm


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else None


def feats(ext):
    s = ext["stances"]
    return {"pol": s[0]["polarity"] if s else None,
            "attr": [x["attribute"]["tag"] for x in s if x["attribute"]],
            "sch": [y for x in s if x["value"] for y in x["value"]["schwartz"] if y != "해당 없음"],
            "ep": len(ext["episodes"]), "n_st": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    a = ap.parse_args()
    d = run_dir(a.run)
    tb = TagBook()
    tags = json.load(open(os.path.join(d, "tags.json"), encoding="utf-8"))
    for k in tb.tags:
        tb.tags[k] = tags.get(k, {})          # 꼬리표 목록은 고정 (자라지 않음)
    llm = get_llm()
    ans = read_jsonl(os.path.join(d, "answers.jsonl"))
    cache_p = os.path.join(d, "compare.jsonl")
    done = {(r["item_id"], r["cond"]): r for r in read_jsonl(cache_p)}
    truth_ext = {}
    from sklearn.feature_extraction.text import TfidfVectorizer
    with open(cache_p, "a", encoding="utf-8") as f:
        for r in ans:
            key = (r["item_id"], r["cond"])
            if key in done:
                continue
            if r["item_id"] not in truth_ext:
                truth_ext[r["item_id"]] = extract_one(llm, r["question"], r["truth"], tb)[0]
            te = truth_ext[r["item_id"]]
            ae = extract_one(llm, r["question"], r["answer"], tb)[0]
            T, A = feats(te), feats(ae)
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([r["truth"], r["answer"]])
            M = v.transform([r["truth"], r["answer"]])
            rec = {"item_id": r["item_id"], "cond": r["cond"],
                   "polarity": (T["pol"] == A["pol"]) if (T["pol"] and A["pol"]) else None,
                   "attr": jac(T["attr"], A["attr"]), "value": jac(T["sch"], A["sch"]),
                   "ep_truth": T["ep"], "ep_twin": A["ep"], "char_sim": float((M[0] @ M[1].T).toarray()[0][0]),
                   "truth_ext": te, "twin_ext": ae}
            done[key] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(r["item_id"], r["cond"], rec["polarity"], rec["attr"], rec["value"])
    agg = defaultdict(lambda: defaultdict(list))
    for r in done.values():
        for k in ["polarity", "attr", "value", "char_sim"]:
            if r[k] is not None:
                agg[r["cond"]][k].append(float(r[k]))
        agg[r["cond"]]["ep_extra"].append(max(0, r["ep_twin"] - r["ep_truth"]))
    mean = lambda xs: f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "—"
    L = ["# 같은 추출기로 비교한 충실도 프로필", "",
         "| 조건 | 찬반 일치 | 특징 겹침 | 가치(Schwartz) 겹침 | 지어낸 일화 수(초과분) | 표면 유사도 |", "|---|---|---|---|---|---|"]
    for c in sorted(agg):
        m = agg[c]
        L.append(f"| {c} | {mean(m['polarity'])} | {mean(m['attr'])} | {mean(m['value'])} | {mean(m['ep_extra'])} | {mean(m['char_sim'])} |")
    L += ["", "B0 = 기본 소개만, B1 = 원문 발췌, O1 = 관계도 근거.",
          "파일럿 숫자는 문항 수가 적어 방향 확인용입니다. 천장(같은 질문에 본인이 다시 답한 Retest)과 함께 봐야 합니다.",
          "찬반 일치는 둘 다 입장이 추출된 문항만 셉니다. n이 작으면 추출기가 입장을 잘 못 뽑는다는 뜻입니다."]
    open(os.path.join(d, "compare.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
