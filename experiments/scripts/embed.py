"""텍스트 유사도 공용 모듈.

metric
  cos        문장 임베딩(e5, mean-pool) 코사인. e5 규칙대로 양쪽에 "query: " 접두어(대칭 유사도).
  bertscore  토큰 임베딩 단위 greedy 정렬 F1 (Zhang et al., 2020, BERTScore).
             문장 하나의 벡터로 뭉개지 않고 토큰끼리 맞춰 보므로 길이·화제 섞임에 덜 둔감하다.
             idf 가중(fit_corpus 기준) + baseline 재스케일((F−b)/(1−b), b = 무관한 쌍의 평균 F).
backend
  st     sentence-transformers (모델: config roles.embedding.model, 기본 intfloat/multilingual-e5-small)
  tfidf  문자 n-gram TF-IDF 코사인 (st 미설치 시 대체, bertscore 불가)

sim(texts_a, texts_b) → 같은 인덱스끼리의 유사도 리스트. dist = 1 − sim.
"""
import json, os, math, random, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = json.load(open(os.path.join(ROOT, "experiments/config/models.json"), encoding="utf-8"))
E5_PREFIX = "query: "


class Embedder:
    def __init__(self, backend="auto", fit_corpus=None, metric="cos", model=None):
        self.backend, self.metric = backend, metric
        self._cache = {}   # text → 벡터(cos) 또는 (토큰 id, 토큰 임베딩)(bertscore). 같은 응답 반복 인코딩 방지
        self.name = model or CFG["roles"]["embedding"]["model"]
        if backend == "auto":
            try:
                import sentence_transformers  # noqa
                self.backend = "st"
            except Exception:
                self.backend = "tfidf"
        if metric == "bertscore" and self.backend != "st":
            raise RuntimeError("bertscore는 sentence-transformers가 필요합니다 (.venv-st 사용)")
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(self.name)
            self.prefix = E5_PREFIX if "e5" in self.name.lower() else ""
            if metric == "bertscore":
                self._fit_bertscore(fit_corpus or [])
                self.name = f"bertscore[{self.name}]"
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.name = "tfidf-char_wb-2_4"
            self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            self.v.fit(fit_corpus or [""])

    # ---------- cos ----------
    def encode(self, texts):
        if self.backend == "st":
            texts = list(texts)
            new = [t for t in dict.fromkeys(texts) if t not in self._cache]
            if new:
                V = self.m.encode([self.prefix + t for t in new], normalize_embeddings=True, batch_size=32)
                self._cache.update(zip(new, np.asarray(V)))
            return np.stack([self._cache[t] for t in texts])
        from sklearn.preprocessing import normalize
        return normalize(self.v.transform(list(texts)))

    # ---------- bertscore ----------
    def _tokens(self, texts):
        """텍스트별 (토큰 id 배열, L2 정규화된 토큰 임베딩) — 특수 토큰 제외."""
        texts = list(texts)
        embs = self.m.encode(texts, output_value="token_embeddings", convert_to_numpy=False)
        tok = self.m.tokenizer(texts, truncation=True, max_length=self.m.max_seq_length)["input_ids"]
        special = set(self.m.tokenizer.all_special_ids)
        out = []
        for ids, e in zip(tok, embs):
            e = e.detach().cpu().numpy()[: len(ids)]
            keep = [i for i, t in enumerate(ids) if t not in special]
            ids = np.asarray(ids)[keep]; e = e[keep]
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            out.append((ids, e))
        return out

    def _fit_bertscore(self, corpus):
        corpus = [c for c in corpus if c and c.strip()]
        # idf: 말뭉치에서 흔한 토큰(조사·어미)일수록 가중 ↓, 미등장 토큰은 최대 idf
        n = len(corpus)
        df = collections.Counter()
        for ids in self.m.tokenizer(corpus, truncation=True, max_length=self.m.max_seq_length)["input_ids"] if n else []:
            df.update(set(ids))
        self.idf_default = math.log(n + 1)
        self.idf = {t: math.log((n + 1) / (c + 1)) for t, c in df.items()}
        # baseline: 서로 다른 텍스트끼리 짝지은 무관 쌍의 평균 F (재스케일 기준점)
        self.baseline = 0.0
        if n >= 10:
            rng = random.Random(0)
            idx = list(range(n)); pairs = []
            for _ in range(min(300, n * (n - 1))):
                i, j = rng.sample(idx, 2); pairs.append((corpus[i], corpus[j]))
            a, b = zip(*pairs)
            self.baseline = float(np.mean(self._bs_raw(a, b)))

    def _w(self, ids):
        w = np.asarray([self.idf.get(int(t), self.idf_default) for t in ids], dtype=float)
        return w if w.sum() > 0 else np.ones(len(ids))

    def _bs_raw(self, a, b):
        new = [t for t in dict.fromkeys(list(a) + list(b)) if t not in self._cache]
        if new:
            self._cache.update(zip(new, self._tokens(new)))
        cache = self._cache
        fs = []
        for x, y in zip(a, b):
            (ix, ex), (iy, ey) = cache[x], cache[y]
            if len(ix) == 0 or len(iy) == 0:
                fs.append(0.0); continue
            S = ex @ ey.T
            wx, wy = self._w(ix), self._w(iy)
            r = float((S.max(1) * wx).sum() / wx.sum())   # recall: x(기준)의 각 토큰이 y에서 얼마나 찾아지나
            p = float((S.max(0) * wy).sum() / wy.sum())   # precision
            fs.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        return fs

    # ---------- 공통 ----------
    def sim(self, a, b):
        a, b = list(a), list(b)
        if self.metric == "bertscore":
            bl = self.baseline
            return [(f - bl) / (1 - bl) for f in self._bs_raw(a, b)]
        A, B = self.encode(a), self.encode(b)
        if self.backend == "st":
            return [float(x) for x in (A * B).sum(1)]
        return [float(A[i].multiply(B[i]).sum()) for i in range(A.shape[0])]
