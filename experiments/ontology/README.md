# 온톨로지 파일럿 (P01 한 사람)

설계 기준: 인물 온톨로지 설계표 v0.3 · 예시 페이지 "꼬리표에서 관계도까지". 저장소 루트에서 실행합니다. 키는 기존과 같이 `experiments/.env`의 `DEEPSEEK_API_KEY`.

## 한 번에

```
# API 없이 형식만 점검
MOCK_LLM=1 python3 experiments/ontology/run_pilot.py --run mock

# 작은 시험: Train 앞 30발화로 관계도, Holdout 20문항으로 답변 비교
python3 experiments/ontology/run_pilot.py --run pilot1 --limit 30 --n 20

# Train 전체(381발화) — 이어서 실행되므로 pilot1을 그대로 늘려도 됨
python3 experiments/ontology/run_pilot.py --run pilot1 --limit 0 --n 50
```

중간에 끊겨도 다시 실행하면 이어서 합니다(LLM 응답은 `experiments/.cache/`에 저장).

## 단계

| 단계 | 스크립트 | 하는 일 | 출력 |
|---|---|---|---|
| ① 추출 | extract.py | 발화를 질문 상황·입장·특징·결과·가치·근거 등급·일화로 분해. 꼬리표 목록은 읽으며 자람 | extract.jsonl, tags.json |
| ② 상자 | build.py | 입장 상자 ST-0001…, 일화 상자 EP-0001…, 공유 점(대상·특징·결과·가치) | graph.json, sheet_*.csv |
| ③ 공리 | check.py | A02·A03·A04·A05·A09·A11·A12·A13·A16·A17·A18 + 구절 원문 일치(G1) | check.md, violations.csv |
| ④ 통계 | stats.py | 칸 채움 비율, 꼬리표 수렴 곡선, 근거 등급, 역량 질문 자동 점검 | stats.md, tag_growth.csv |
| ⑤ 답변 | answer.py | Holdout 질문에 B0(소개만)·B1(원문 발췌)·O1(관계도 근거)로 트윈 답 생성 | answers.jsonl, prompts_sample.md |
| ⑥ 비교 | compare.py | 실제 답과 트윈 답을 같은 추출기로 분해해 찬반·특징·가치·일화를 맞댐 | compare.md |

결과는 `experiments/runs/ontology/<run>/`에 쌓입니다.

## 사람이 볼 것

1. `sheet_코딩.csv`, `sheet_꼬리표.csv` — 코딩 시트와 같은 열 순서. 모델 초안을 시트에서 검토하고, 꼬리표를 합치거나 지운 뒤 '꼬리표' 탭을 CSV로 내려받아 `--tags`로 다시 돌리면 그 목록에서 시작합니다.
2. `stats.md` — 칸이 너무 비는지(특히 특징 A), 꼬리표가 수렴하는지.
3. `check.md` — G1(구절을 지어냄)이 많으면 추출 프롬프트(common.py)를 고칩니다.
4. `compare.md` — O1이 B1보다 나은지, 일화를 지어내는지.

정답 비교용으로 발화 10~15개는 모델 초안을 보기 전에 직접 코딩해 두세요(코딩 시트 '일치도' 탭).

## 비용(대략)

DeepSeek 기준: 30발화 시험은 호출 약 30 + 60 + 80회, Train 전체 + Holdout 50은 약 380 + 150 + 200회.

## 한계

- 한 사람 자료라 A13(2명 이상 공유)과 비슷한 사람 빌리기(조건 D·E)는 아직 시험할 수 없습니다.
- 특질·동기·지향·상황별 방침 상자는 이번 파일럿에서 뽑지 않습니다(입장·일화만).
- 비교의 찬반 일치는 '첫 입장'끼리만 봅니다. 입장이 여러 개인 답은 손으로 확인이 필요합니다.
