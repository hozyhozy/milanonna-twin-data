#!/usr/bin/env python3
"""⑤-1 생성 호출. holdout50 × 조건(G0/G1/G2/G3) × 형식(F1/F2) × 온도 × 반복 → runs/<exp>/<cond>.jsonl
(G3는 실명이 들어가므로 runs_local/<exp>/G3.jsonl 에 저장.)

드라이런(비용·형식 확인):  python3 experiments/scripts/run.py --exp A1 --dry
A1 본실행:                 python3 experiments/scripts/run.py --exp A1
A3(온도):                  python3 experiments/scripts/run.py --exp A3 --conds G2 --formats F1 --temps 0,0.7,1
D4(CoT):                   python3 experiments/scripts/run.py --exp D4 --conds G2 --formats F1 --cot
같은 (프롬프트,온도,rep) 호출은 캐시되므로 중단 후 재실행해도 이어서 진행된다.
"""
import json, os, sys, argparse, hashlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import LLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = lambda *a: os.path.join(ROOT, *a)
CFG = json.load(open(P("experiments/config/models.json"), encoding="utf-8"))
COT_SUFFIX = "\n\n먼저 노라의 입장에서 그렇게 생각하는 이유를 3문장으로 쓰고, 마지막 줄에 답만 따로 쓰세요."


def load_prompts():
    pr = {}
    for n in ["system_G0", "system_G1", "system_G2", "user_F1", "user_F2"]:
        pr[n] = open(P("experiments/prompts", n + ".txt"), encoding="utf-8").read().strip()
    g3 = P("experiments/prompts/system_G3.txt")
    pr["system_G3"] = open(g3, encoding="utf-8").read().strip() if os.path.exists(g3) else None
    return pr


def build_system(cond, item, pr, retrieved, train_by_id):
    if cond == "G0":
        return pr["system_G0"]
    if cond == "G1":
        return pr["system_G1"]
    if cond == "G3":
        return pr["system_G3"]
    if cond == "G2":
        block = []
        for r in retrieved[item["item_id"]]:
            t = train_by_id[r["id"]]
            block.append(f"Q: {t['question']}\nA: {t['answer']}")
        return pr["system_G2"].replace("{G1}", pr["system_G1"]).replace("{RETRIEVED}", "\n\n".join(block))
    raise ValueError(cond)


def build_user(fmt, item, pr, cot):
    if fmt == "F1":
        u = pr["user_F1"].replace("{question}", item["question"])
    else:
        if not item.get("stance_truth"):
            return None
        u = pr["user_F2"].replace("{question}", item["question"]).replace("{stance}", item["stance_truth"])
    return u + COT_SUFFIX if cot else u


def parse(fmt, text):
    if fmt == "F1":
        return text.strip()
    m = re.findall(r"[1-7]", text.strip().splitlines()[-1] if text.strip() else "")
    return int(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="A1")
    ap.add_argument("--conds", default="G0,G1,G2,G3")
    ap.add_argument("--formats", default="F1,F2")
    ap.add_argument("--temps", default="1")
    ap.add_argument("--reps", type=int, default=20, help="T>0 반복 수")
    ap.add_argument("--no-t0", action="store_true", help="T=0 1회를 추가하지 않음")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--items", type=int, default=0, help="앞에서 N문항만")
    ap.add_argument("--dry", action="store_true", help="5문항 × 2회 × T=1 만")
    a = ap.parse_args()
    if a.dry:
        a.items, a.reps, a.no_t0 = 5, 2, True

    pr = load_prompts()
    items = json.load(open(P("experiments/items/holdout50.json"), encoding="utf-8"))["items"]
    if a.items:
        items = items[: a.items]
    retrieved = json.load(open(P("experiments/items/retrieved_ids.json"), encoding="utf-8"))["retrieved"]
    train_by_id = {t["id"]: t for t in json.load(open(P("data/anon/Train.json"), encoding="utf-8"))["items"]}
    llm = LLM(role="generation")
    temps = [float(x) for x in a.temps.split(",")]
    plan = [(t, r) for t in temps for r in range(a.reps if t > 0 else 1)]
    if not a.no_t0 and 0.0 not in temps:
        plan = [(0.0, 0)] + plan

    for cond in a.conds.split(","):
        if cond == "G3" and not pr["system_G3"]:
            print("[건너뜀] G3: experiments/prompts/system_G3.txt 없음 (로컬 파일을 복사하세요)"); continue
        base = "experiments/runs_local" if cond == "G3" else "experiments/runs"
        os.makedirs(P(base, a.exp), exist_ok=True)
        out = P(base, a.exp, f"{cond}{'_cot' if a.cot else ''}.jsonl")
        done = set()
        if os.path.exists(out):
            for line in open(out, encoding="utf-8"):
                j = json.loads(line); done.add((j["item_id"], j["format"], j["temp"], j["rep"]))
        n_new = 0; tok = 0
        with open(out, "a", encoding="utf-8") as f:
            for it in items:
                system = build_system(cond, it, pr, retrieved, train_by_id)
                for fmt in a.formats.split(","):
                    user = build_user(fmt, it, pr, a.cot)
                    if user is None:
                        continue
                    for temp, rep in plan:
                        key = (it["item_id"], fmt, temp, rep)
                        if key in done:
                            continue
                        text, meta = llm.chat(system, user, temperature=temp, seed=rep,
                                              max_tokens=400 if fmt == "F1" or a.cot else 8)
                        rec = {"exp": a.exp, "item_id": it["item_id"], "stratum": it["stratum"], "cond": cond,
                               "format": fmt, "cot": a.cot, "temp": temp, "rep": rep,
                               "model": meta["model"], "retrieved_ids": [r["id"] for r in retrieved[it["item_id"]]] if cond == "G2" else [],
                               "prompt_hash": meta["prompt_hash"], "response": text, "parsed": parse(fmt, text),
                               "tokens_in": meta["tokens_in"], "tokens_out": meta["tokens_out"], "cached": meta["cached"], "ts": meta["ts"]}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                        n_new += 1; tok += (meta["tokens_in"] or 0) + (meta["tokens_out"] or 0)
        print(f"{cond}: 신규 {n_new}건 (누적 {len(done) + n_new}), 토큰 {tok:,} → {out}")


if __name__ == "__main__":
    main()
