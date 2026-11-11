#!/usr/bin/env python3
"""⑤-2 채점. runs/<exp>/*.jsonl (+ runs_local) 의 F1 응답마다
  (1) 판정 모델 눈가림 A/B 채점 → stance/ground/valence/voice (0~100)
  (2) 임베딩 유사도 sim_emb (응답 ↔ 본인 답)
  (3) --tag: 응답 자체의 valence/ground/일화 태깅 (A2·C1용)
  (4) --retest: Retest 14쌍(시즌1 Train 답 vs 시즌2 답)을 같은 판정에 넣어 본인 천장 계산
  (5) --swap: A/B 순서를 바꿔 한 번 더 채점 → 판정 자기 일관성
결과: runs/<exp>/judge_<cond>.jsonl, results/<exp>_retest_ceiling.json
G3 응답은 채점 전에 가명 사전으로 치환(조건 눈가림).

python3 experiments/scripts/judge.py --exp A1 [--tag] [--retest] [--swap]
"""
import json, os, sys, argparse, glob, random, re, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import LLM
from embed import Embedder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = lambda *a: os.path.join(ROOT, *a)
JUDGE = open(P("experiments/prompts/judge.txt"), encoding="utf-8").read()
TAG = open(P("experiments/prompts/tag_items.txt"), encoding="utf-8").read()
TAG_RESP = TAG + "\n추가로 \"anecdotes\": 답변에 나오는 구체적 일화(시점·장소·인물이 있는 사건)를 짧은 구로 나열한 리스트(없으면 [])."


def load_pseudo():
    p = P("experiments/config/pseudonym_dict.json")
    if not os.path.exists(p):
        return lambda s: s
    d = json.load(open(p, encoding="utf-8"))
    rules = [(unicodedata.normalize("NFC", a), b) for a, b in d["rules"]]
    rx = [(re.compile(a), b) for a, b in d.get("regex_rules", [])]

    def f(s):
        s = unicodedata.normalize("NFC", s)
        for pat, rep in rx:
            s = pat.sub(rep, s)
        for a, b in rules:
            s = s.replace(a, b)
        return s
    return f


def judge_pair(llm, question, truth, resp, seed, swap=False):
    rng = random.Random(f"{seed}{int(swap)}")
    a_is_truth = rng.random() < 0.5
    if swap:
        a_is_truth = not a_is_truth
    a, b = (truth, resp) if a_is_truth else (resp, truth)
    text, meta = llm.chat("당신은 두 텍스트의 일치도를 채점하는 평가자입니다. JSON만 출력합니다.",
                          JUDGE.replace("{question}", question).replace("{a}", a).replace("{b}", b),
                          temperature=0, json_mode=True, seed=f"judge{int(swap)}", max_tokens=120)
    try:
        j = json.loads(text)
        return {k: float(j[k]) for k in ["stance", "ground", "valence", "voice"]}, a_is_truth, meta
    except Exception:
        return None, a_is_truth, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="A1")
    ap.add_argument("--tag", action="store_true"); ap.add_argument("--retest", action="store_true")
    ap.add_argument("--swap", action="store_true"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    items = {x["item_id"]: x for x in json.load(open(P("experiments/items/holdout50.json"), encoding="utf-8"))["items"]}
    pseudo = load_pseudo()
    llm = LLM(role="judge"); tagger = LLM(role="tagger") if a.tag else None
    files = glob.glob(P("experiments/runs", a.exp, "*.jsonl")) + glob.glob(P("experiments/runs_local", a.exp, "*.jsonl"))
    files = [f for f in files if not os.path.basename(f).startswith("judge_")]
    corpus = [x["answer_truth"] for x in items.values()]
    emb = Embedder(a.backend, fit_corpus=corpus)
    print("embedding:", emb.backend, emb.name)

    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        recs = [r for r in recs if r["format"] == "F1"]
        if not recs:
            continue
        out = os.path.join(os.path.dirname(f), "judge_" + os.path.basename(f))
        done = set()
        if os.path.exists(out):
            done = {(json.loads(l)["item_id"], json.loads(l)["temp"], json.loads(l)["rep"]) for l in open(out, encoding="utf-8")}
        todo = [r for r in recs if (r["item_id"], r["temp"], r["rep"]) not in done]
        if not todo:
            print(f"{os.path.basename(f)}: 완료"); continue
        truths = [items[r["item_id"]]["answer_truth"] for r in todo]
        resps = [pseudo(r["response"]) if r["cond"] == "G3" else r["response"] for r in todo]
        sims = emb.sim(resps, truths)
        def work(args):
            r, resp, s = args
            it = items[r["item_id"]]
            seed = f"{r['item_id']}|{r['cond']}|{r['temp']}|{r['rep']}"
            sc, a_truth, meta = judge_pair(llm, it["question"], it["answer_truth"], resp, seed)
            rec = {**{k: r[k] for k in ["exp", "item_id", "stratum", "cond", "format", "cot", "temp", "rep"]},
                   "sim_emb": round(s, 4), "emb_model": emb.name, "judge_model": meta["model"],
                   "truth_was_A": a_truth, "scores": sc}
            if a.swap:
                sc2, _, _ = judge_pair(llm, it["question"], it["answer_truth"], resp, seed, swap=True)
                rec["scores_swap"] = sc2
            if tagger:
                t, _ = tagger.chat("당신은 텍스트 코딩을 돕는 연구 보조원입니다. JSON만 출력합니다.",
                                   TAG_RESP.replace("{question}", it["question"]).replace("{answer}", resp),
                                   temperature=0, json_mode=True, seed="tag", max_tokens=300)
                try:
                    rec["resp_tags"] = json.loads(t)
                except Exception:
                    rec["resp_tags"] = None
            return rec

        from concurrent.futures import ThreadPoolExecutor, as_completed
        n = 0
        with open(out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, x) for x in zip(todo, resps, sims)]
            for fu in as_completed(futs):
                try:
                    rec = fu.result()
                except Exception as e:
                    print("  [실패]", e); continue
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
                n += 1
                if n % 100 == 0:
                    print(f"  {os.path.basename(f)}: {n}/{len(todo)}")
        print(f"{os.path.basename(f)}: {len(todo)}건 채점 → {out}")

    if a.retest:
        rt = json.load(open(P("data/anon/Retest.json"), encoding="utf-8"))["items"]
        tr = json.load(open(P("data/anon/Train.json"), encoding="utf-8"))["items"]
        by_group = {}
        for t in tr:
            if t.get("dup_group"):
                by_group.setdefault(t["dup_group"], []).append(t)
        res = []
        for r in rt:
            for t in by_group.get(r["dup_group"], []):
                sc, _, _ = judge_pair(llm, r["question"], r["answer"], t["answer"], f"retest|{r['id']}|{t['id']}")
                s = emb.sim([r["answer"]], [t["answer"]])[0]
                res.append({"retest_id": r["id"], "train_id": t["id"], "dup_group": r["dup_group"], "sim_emb": round(s, 4), "scores": sc})
        os.makedirs(P("experiments/results"), exist_ok=True)
        json.dump(res, open(P("experiments/results", f"{a.exp}_retest_ceiling.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        ok = [x for x in res if x["scores"]]
        if ok:
            print(f"Retest 천장: {len(ok)}쌍, stance 평균 {sum(x['scores']['stance'] for x in ok)/len(ok):.1f}, sim_emb 평균 {sum(x['sim_emb'] for x in ok)/len(ok):.3f}")


if __name__ == "__main__":
    main()
