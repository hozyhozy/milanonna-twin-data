"""OpenAI 호환 chat API 공용 클라이언트 (DeepSeek 기본). 캐시·재시도·로그 포함.

사용:
    from llm import LLM
    llm = LLM(role="generation")                 # config/models.json의 roles에서 provider/model 선택
    text, meta = llm.chat(system="...", user="...", temperature=1.0, seed=7)
meta = {"model","provider","tokens_in","tokens_out","cached","prompt_hash","ts"}
"""
import hashlib, json, os, time, datetime
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = json.load(open(os.path.join(ROOT, "experiments/config/models.json"), encoding="utf-8"))


def _load_env():
    p = os.path.join(ROOT, "experiments/.env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class LLM:
    def __init__(self, role="generation", provider=None, model=None, use_cache=True):
        _load_env()
        r = CFG["roles"][role]
        self.provider = provider or r["provider"]
        self.model = model or r["model"]
        p = CFG["providers"][self.provider]
        self.base_url = p["base_url"].rstrip("/")
        self.key = os.environ.get(p["key_env"])
        self.mock = os.environ.get("MOCK_LLM") == "1"   # MOCK_LLM=1 이면 API 없이 가짜 응답(파이프라인 점검용)
        if not self.key and not self.mock:
            raise RuntimeError(f"{p['key_env']} 가 없습니다. experiments/.env 에 넣으세요 (.env.example 참고)")
        self.d = CFG["defaults"]
        self.cache_dir = os.path.join(ROOT, self.d["cache_dir"]) if use_cache else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _mock(self, system, user, json_mode, seed):
        import random
        r = random.Random(str(seed) + user[:40])
        if json_mode and "anecdotes" in user:
            return json.dumps({"valence": r.randint(-2, 2), "ground": r.choice(["일화", "원칙", "사실", "혼합"]), "stance": "모의 진술", "stance_score": r.randint(4, 7), "anecdotes": []}, ensure_ascii=False)
        if json_mode and "stance_score" in user:
            return json.dumps({"valence": r.randint(-2, 2), "valence_reason": "mock", "ground": r.choice(["일화", "원칙"]), "ground_reason": "mock", "stance": "모의 진술문입니다.", "stance_score": r.randint(5, 7)}, ensure_ascii=False)
        if json_mode:
            return json.dumps({k: r.randint(30, 95) for k in ["stance", "ground", "valence", "voice"]})
        if "숫자 하나만" in user:
            return str(r.randint(1, 7))
        return f"(모의 응답 {seed}) 나는 그렇게 생각해요. 옛날에 그런 일이 있었거든. {r.choice(['친구들', '할머니', '지금을 즐겨'])}."

    def chat(self, system, user, temperature=1.0, max_tokens=None, seed=None, json_mode=False):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature, "max_tokens": max_tokens or self.d["max_tokens"]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.mock:
            return self._mock(system, user, json_mode, seed), {"model": "mock", "provider": "mock", "tokens_in": 0,
                "tokens_out": 0, "cached": False, "prompt_hash": "mock", "ts": datetime.datetime.now().isoformat(timespec="seconds")}
        # 캐시 키: 같은 프롬프트·온도라도 seed(=반복 인덱스)가 다르면 다른 호출로 취급
        h = hashlib.sha256(json.dumps([body, seed], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
        cp = os.path.join(self.cache_dir, h + ".json") if self.cache_dir else None
        if cp and os.path.exists(cp):
            c = json.load(open(cp, encoding="utf-8"))
            c["meta"]["cached"] = True
            return c["text"], c["meta"]
        last = None
        for i in range(self.d["retries"]):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.d["timeout_s"],
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                text = j["choices"][0]["message"]["content"]
                u = j.get("usage", {})
                meta = {"model": self.model, "provider": self.provider, "tokens_in": u.get("prompt_tokens"),
                        "tokens_out": u.get("completion_tokens"), "cached": False, "prompt_hash": h,
                        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
                if cp:
                    json.dump({"text": text, "meta": meta, "body": body, "seed": seed}, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
                return text, meta
            except Exception as e:  # noqa
                last = e
                time.sleep(2 ** i)
        raise RuntimeError(f"LLM 호출 실패: {last}")
