# experiments — 파일럿 실행 코드

설계: `docs/파일럿_실험설계_편향테스트_v1.md`. 모든 스크립트는 저장소 루트에서 실행.

```
cp experiments/.env.example experiments/.env   # DEEPSEEK_API_KEY 입력
python3 experiments/scripts/pseudonymize.py    # ① 가명본 → data/anon/ (사전: experiments/config/pseudonym_dict.json, 로컬)
python3 experiments/scripts/select_items.py    # ② 50문항 층화 추출 → experiments/items/holdout50.json
python3 experiments/scripts/tag_items.py       # ② 판정 모델 1차 태깅 → json + tagging_log.md, 이후 직접 검수(reviewed=true, fame 입력)
cp ~/Desktop/밀라논나/system_G3.txt experiments/prompts/   # ③ G3 실명 프롬프트(로컬 전용, .gitignore)
python3 experiments/scripts/retrieve.py [--backend st]   # ④ 발췌 고정 → items/retrieved_ids.json
python3 experiments/scripts/run.py --exp A1 --dry        # ⑤ 드라이런: 5문항×4조건×2형식×2회 (약 80회 호출)
python3 experiments/scripts/run.py --exp A1              #    본실행: 50×4×2×21 = 8,400회 (중단해도 캐시로 이어감)
python3 experiments/scripts/judge.py --exp A1 --tag --retest --swap   # 채점 + 응답 태깅 + Retest 천장 + 자기일관성
python3 experiments/scripts/analyze.py --exp A1          # → results/A1_summary.md / .csv
```

API 없이 파이프라인만 점검: 모든 명령 앞에 `MOCK_LLM=1` (가짜 응답). 예: `MOCK_LLM=1 python3 experiments/scripts/run.py --exp TEST --dry`

다른 실험: `run.py --exp A3 --conds G2 --formats F1 --temps 0,0.7,1` / `--exp D4 --conds G2 --formats F1 --cot` 등. judge/analyze는 `--exp` 이름만 맞추면 된다.

- `config/models.json`: 역할(generation/judge/tagger/embedding)별 provider·model. DeepSeek 기본.
- `scripts/llm.py`: OpenAI 호환 클라이언트. 같은 (프롬프트, 온도, seed) 호출은 `.cache/`에서 재사용 → 중단 후 재실행해도 과금 중복 없음.
- `judge.py` 출력 `judge_<cond>.jsonl`: `scores`(stance/ground/valence/voice 0~100), `sim_emb`, `truth_was_A`(눈가림 순서), `scores_swap`(--swap), `resp_tags`(--tag: 응답의 valence/ground/anecdotes).
- `analyze.py`가 만드는 표: 조건별 일치도(천장 대비), 층·유명·엄격 분해, 거리 삼각형, 반복 응답 분산, A2/C1, B1 충돌 집합·끌림 지수, E1 leak/순기여, F2 acc.
- `items/holdout50.json`의 `stratum`은 휴리스틱 분류(키워드)이므로 연구자가 보고 고칠 것. 교체 시 `item_id`만 Holdout의 다른 문항으로 바꾸면 됨.
