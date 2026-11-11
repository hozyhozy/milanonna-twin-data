# experiments — 파일럿 실행 코드

설계: `docs/파일럿_실험설계_편향테스트_v1.md`. 모든 스크립트는 저장소 루트에서 실행.

```
cp experiments/.env.example experiments/.env   # DEEPSEEK_API_KEY 입력
python3 experiments/scripts/pseudonymize.py    # ① 가명본 → data/anon/ (사전: experiments/config/pseudonym_dict.json, 로컬)
python3 experiments/scripts/select_items.py    # ② 50문항 층화 추출 → experiments/items/holdout50.json
python3 experiments/scripts/tag_items.py       # ② 판정 모델 1차 태깅 → json + tagging_log.md, 이후 직접 검수(reviewed=true, fame 입력)
# ③ prompts/system_G1.txt (800자 요약), system_G3.txt(실명, 로컬)
# ④ retrieve.py → items/retrieved_ids.json
# ⑤ run.py / judge.py / analyze.py
```

- `config/models.json`: 역할(generation/judge/tagger/embedding)별 provider·model. DeepSeek 기본.
- `scripts/llm.py`: OpenAI 호환 클라이언트. 같은 (프롬프트, 온도, seed) 호출은 `.cache/`에서 재사용 → 중단 후 재실행해도 과금 중복 없음.
- `items/holdout50.json`의 `stratum`은 휴리스틱 분류(키워드)이므로 연구자가 보고 고칠 것. 교체 시 `item_id`만 Holdout의 다른 문항으로 바꾸면 됨.
