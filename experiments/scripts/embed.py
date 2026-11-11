"""텍스트 유사도 공용 모듈. sentence-transformers(e5) 가능하면 사용, 아니면 TF-IDF 문자 n-gram.
sim(texts_a, texts_b) → 같은 인덱스끼리의 코사인 유사도 리스트."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = json.load(open(os.path.join(ROOT, "experiments/config/models.json"), encoding="utf-8"))


class Embedder:
    def __init__(self, backend="auto", fit_corpus=None):
        self.backend = backend
        self.name = CFG["roles"]["embedding"]["model"]
        if backend == "auto":
            try:
                import sentence_transformers  # noqa
                self.backend = "st"
            except Exception:
                self.backend = "tfidf"
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(self.name)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.name = "tfidf-char_wb-2_4"
            self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            self.v.fit(fit_corpus or [""])

    def encode(self, texts):
        if self.backend == "st":
            return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))
        X = self.v.transform(list(texts))
        from sklearn.preprocessing import normalize
        return normalize(X)

    def sim(self, a, b):
        A, B = self.encode(a), self.encode(b)
        if self.backend == "st":
            return [float(x) for x in (A * B).sum(1)]
        return [float(A[i].multiply(B[i]).sum()) for i in range(A.shape[0])]
