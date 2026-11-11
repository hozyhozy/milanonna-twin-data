#!/usr/bin/env python3
"""④ 발췌 고정: holdout50 각 문항에 대해 Train(가명본) 381문항 중 유사한 k개를 골라
experiments/items/retrieved_ids.json 에 저장한다. run.py는 실행 시 검색하지 않고 이 파일을 읽는다(재현성).

backend:
  st     sentence-transformers (config/models.json roles.embedding.model, 기본 intfloat/multilingual-e5-small)
  tfidf  scikit-learn 문자 n-gram TF-IDF (의존성 가벼움, 모델 다운로드 불필요)
  기본값 auto: st 가능하면 st, 아니면 tfidf.
질의 = Holdout 질문, 문서 = Train 질문 + 답변 앞 200자 (질문만 쓰면 derived 질문끼리 표면 유사도에 끌림).

python3 experiments/scripts/retrieve.py --k 20 [--backend tfidf]
"""
import json, os, argparse, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = json.load(open(os.path.join(ROOT, "experiments/config/models.json"), encoding="utf-8"))


def doc_text(x):
    return x["question"] + " " + x["answer"][:200]


def embed_st(queries, docs, model_name):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer(model_name)
    q = m.encode(["query: " + t for t in queries], normalize_embeddings=True)
    d = m.encode(["passage: " + t for t in docs], normalize_embeddings=True)
    return np.asarray(q) @ np.asarray(d).T


def embed_tfidf(queries, docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    D = v.fit_transform(docs)
    Q = v.transform(queries)
    return (Q @ D.T).toarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--backend", default="auto", choices=["auto", "st", "tfidf"])
    ap.add_argument("--items", default=os.path.join(ROOT, "experiments/items/holdout50.json"))
    ap.add_argument("--train", default=os.path.join(ROOT, "data/anon/Train.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments/items/retrieved_ids.json"))
    a = ap.parse_args()

    items = json.load(open(a.items, encoding="utf-8"))["items"]
    train = json.load(open(a.train, encoding="utf-8"))["items"]
    queries = [x["question"] for x in items]
    docs = [doc_text(x) for x in train]

    backend = a.backend
    model_name = CFG["roles"]["embedding"]["model"]
    if backend == "auto":
        try:
            import sentence_transformers  # noqa
            backend = "st"
        except Exception:
            backend = "tfidf"
    if backend == "st":
        try:
            S = embed_st(queries, docs, model_name)
        except Exception as e:
            print(f"[경고] sentence-transformers 실패({e.__class__.__name__}) → tfidf로 대체")
            backend = "tfidf"
    if backend == "tfidf":
        S = embed_tfidf(queries, docs)
        model_name = "tfidf-char_wb-2_4"

    out = {}
    for i, x in enumerate(items):
        order = S[i].argsort()[::-1][: a.k]
        out[x["item_id"]] = [{"id": train[j]["id"], "score": round(float(S[i][j]), 4)} for j in order]
    train_hash = hashlib.sha256(json.dumps([t["id"] for t in train]).encode()).hexdigest()[:12]
    json.dump({"backend": backend, "model": model_name, "k": a.k, "doc": "question + answer[:200]",
               "train_hash": train_hash, "retrieved": out},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 간단 점검: 상위 1개 미리보기
    by_id = {t["id"]: t for t in train}
    for x in items[:3]:
        top = out[x["item_id"]][0]
        print(f"- {x['question'][:40]}  →  {by_id[top['id']]['question'][:40]} ({top['score']})")
    print(f"backend={backend} model={model_name} k={a.k} → {a.out}")


if __name__ == "__main__":
    main()
