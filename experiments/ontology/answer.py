#!/usr/bin/env python3
"""⑤ 새 질문에 답하기: Holdout 질문에 세 조건으로 트윈 답을 만든다.

  B0  기본 소개만 (빈 LLM에 가까운 바닥)
  B1  원문 발췌 — Train 질문·답 중 비슷한 k개 (기존 G2 방식, 비교 기준)
  O1  관계도 근거 — 비슷한 입장 상자 k개 + 본인 일화 + 자주 지키는 가치 요약 (이 연구의 방식)

python3 experiments/ontology/answer.py --run pilot1 --n 30 --k 8
출력: answers.jsonl, prompts_sample.md
"""
import argparse, json, os
from collections import Counter
from common import ROOT, run_dir, load_split, PERSON_NAME

RULE = "\n\n[답하는 법]\n아래 근거를 참고하되 질문에 맞게 새로 답하세요. 근거에 없는 구체적인 일화(때·곳·사건)는 지어내지 마세요."


def tfidf_rank(queries, docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    D = v.fit_transform(docs)
    return (v.transform(queries) @ D.T).toarray()


def box_text(b, g):
    q = g["statements"][b["evidencedBy"][0]]["question"]
    if b["type"] == "Episode":
        return f"{q} {b['label']} {b['phrase']}"
    return " ".join(str(x) for x in [q, b["hasTopic"], b["hasAttribute"], b["hasConsequence"], b["hasValue"],
                                     b["phrases"]["topic"]] if x)


def box_line(b):
    if b["type"] == "Episode":
        return f"- [{b['id']} 일화] {b['label'].split(' · ', 1)[1]}: \"{b['phrase']}\""
    v = f"{b['hasValue']}({', '.join(b['schwartz'])}, {b['valueEffect']})" if b["hasValue"] else "-"
    return (f"- [{b['id']}] {b['hasTopic']} → {b['hasPolarity']} | 특징: {b['hasAttribute'] or '-'} | "
            f"결과: {b['hasConsequence'] or '-'} | 가치: {v} | 원문: \"{(b['phrases']['topic'] or '')[:80]}\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--n", type=int, default=30, help="Holdout 문항 수 (holdout50.json 앞에서부터)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conds", default="B0,B1,O1")
    a = ap.parse_args()
    d = run_dir(a.run)
    g = json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))
    items = json.load(open(os.path.join(ROOT, "experiments/items/holdout50.json"), encoding="utf-8"))["items"][:a.n]
    intro = open(os.path.join(ROOT, "experiments/prompts/system_G0.txt"), encoding="utf-8").read().strip()
    qs = [x["question"] for x in items]

    # B1: Train 원문 발췌
    train = load_split("Train")
    sim_t = tfidf_rank(qs, [x["question"] + " " + x["answer"][:200] for x in train])
    # O1: 관계도 상자
    boxes = g["boxes"]
    stance = [b for b in boxes if b["type"] == "Stance"]
    eps = [b for b in boxes if b["type"] == "Episode"]
    sim_s = tfidf_rank(qs, [box_text(b, g) for b in stance]) if stance else None
    sim_e = tfidf_rank(qs, [box_text(b, g) for b in eps]) if eps else None
    sch = Counter(x for b in stance for x in b["schwartz"] if x != "해당 없음").most_common(4)
    styles = Counter(s["style"] for s in g["statements"].values() if s["style"]).most_common(3)
    profile = ("[자주 지키는 가치] " + ", ".join(f"{k}({v})" for k, v in sch) +
               "\n[말투 메모] " + " / ".join(s for s, _ in styles))

    if os.environ.get("MOCK_LLM") == "1":
        llm = None
    else:
        from llm import LLM
        llm = LLM(role="generation")
    out_p = os.path.join(d, "answers.jsonl")
    done = {(r["item_id"], r["cond"]) for r in map(json.loads, open(out_p, encoding="utf-8"))} if os.path.exists(out_p) else set()
    sample = []
    with open(out_p, "a", encoding="utf-8") as f:
        for i, x in enumerate(items):
            ctx = {}
            ctx["B0"] = intro
            top = sim_t[i].argsort()[::-1][:a.k]
            ctx["B1"] = intro + f"\n\n[{PERSON_NAME}가 과거 인터뷰에서 한 말 — 참고]\n" + "\n".join(
                f"- 질문: {train[j]['question']} / 답: {train[j]['answer'][:250]}" for j in top) + RULE
            lines = []
            if sim_s is not None:
                lines += [box_line(stance[j]) for j in sim_s[i].argsort()[::-1][:a.k]]
            if sim_e is not None:
                lines += [box_line(eps[j]) for j in sim_e[i].argsort()[::-1][:3]]
            ctx["O1"] = intro + f"\n\n[{PERSON_NAME}의 관계도에서 꺼낸 근거 — 입장 상자와 본인 일화]\n" + "\n".join(lines) + "\n\n" + profile + RULE
            for c in a.conds.split(","):
                if (x["item_id"], c) in done:
                    continue
                if llm is None:
                    text = f"(모의 {c}) 나는 그렇게 생각해요. 오래 쓰는 게 좋아요."
                else:
                    text, _ = llm.chat(system=ctx[c], user=x["question"], temperature=0.7, seed=1)
                f.write(json.dumps({"item_id": x["item_id"], "cond": c, "question": x["question"],
                                    "truth": x["answer_truth"], "answer": text.strip()}, ensure_ascii=False) + "\n")
                f.flush()
            if i < 2:
                sample.append(f"## {x['item_id']} — {x['question']}\n\n### O1 시스템 프롬프트\n```\n{ctx['O1']}\n```\n\n### B1 시스템 프롬프트\n```\n{ctx['B1']}\n```\n")
            print(f"[{i+1}/{len(items)}] {x['item_id']}")
    open(os.path.join(d, "prompts_sample.md"), "w", encoding="utf-8").write("\n".join(sample))
    print(f"완료 → {out_p}")


if __name__ == "__main__":
    main()
