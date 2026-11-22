# 온톨로지 파일럿 (P01 한 사람)

설계 기준: 인물 온톨로지 설계표 v0.3 · 예시 페이지 "꼬리표에서 관계도까지". 저장소 루트에서 실행합니다. 키는 기존과 같이 `experiments/.env`의 `DEEPSEEK_API_KEY`.

## v2 (2026-11-22) — 1차 결과 반영

| 바뀐 것 | 이유 (pilot1) |
|---|---|
| A·C·V 경계: 계기 사건 → 일화, 본인 성격 → 특질 단서(trait_cues), 말 안 했으면 비움 | A 칸에 '노래에 반함', '책임감' 같은 계기·성격이 들어감 |
| 가치 칸을 정리된 말(words)과 원문 구절(phrase)로 분리 | 자막 오류 문장이 가치 꼬리표가 됨 |
| 근거 등급을 찬반·가치에 따로 | 98%가 '직접 진술'로 나와 구분이 안 됨 |
| 만든 질문(derived)은 질문 상황 '알 수 없음' | 원래 질문이 없어 상황을 알 수 없음 |
| 말투 메모에서 자막 인식 오류 제외 | 트윈이 '예 아,' 같은 인식 오류를 말투로 흉내 냄 |
| clean.py: 자막 교정(선택, 원문은 보존) | 자막 오류 |
| merge.py: 꼬리표 합치기(코드북) | 꼬리표가 수렴하지 않음(한 번만 쓰인 것 74~88%) |
| compare.py: 대상이 같은 입장끼리 찬반 비교, 일화를 '본인 일화 재사용'과 '근거 없는 일화'로 구분, Retest 천장 | 첫 입장끼리만 비교, 일화 지표가 재사용과 지어냄을 구분 못 함 |

### v2 실행 순서

```
# 1) 교정 + 추출 + 비교 (pilot1과 같은 30발화·20문항)
.venv/bin/python experiments/ontology/run_pilot.py --run pilot2 --limit 30 --n 20 --clean

# 2) 꼬리표 합치기 제안 → CSV를 열어 고침(묶음 쪼개기·합치기, 대표 이름, 상태)
.venv/bin/python experiments/ontology/merge.py --run pilot2

# 3) 고친 CSV를 적용해 새 run으로 → 관계도·공리·통계 다시
.venv/bin/python experiments/ontology/merge.py --run pilot2 --apply experiments/runs/ontology/pilot2/merge_proposal.csv --to pilot2m
.venv/bin/python experiments/ontology/run_pilot.py --run pilot2m --stages build,check,stats

# 4) 다음 추출부터는 확정한 코드북에서 고르게
.venv/bin/python experiments/ontology/run_pilot.py --run pilot3 --limit 0 --n 50 --clean --tags experiments/runs/ontology/pilot2/merge_proposal.csv
```

교정본은 `experiments/runs/ontology/_clean/`에 한 번만 만들어 모든 run이 같이 씁니다. 유사도가 0.75 미만인 교정은 '고쳐 쓴 것'으로 보고 버립니다(원문 사용).

## 한 번에 (v1)

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
| ⓪ 교정 (선택) | clean.py | 자동 자막의 인식 오류만 최소로 고침 | _clean/<split>.jsonl |
| ① 추출 | extract.py | 발화를 질문 상황·입장·특징·결과·가치·근거 등급·일화로 분해. 꼬리표 목록은 읽으며 자람 | extract.jsonl, tags.json |
| ② 상자 | build.py | 입장 상자 ST-0001…, 일화 상자 EP-0001…, 공유 점(대상·특징·결과·가치) | graph.json, sheet_*.csv |
| ③ 공리 | check.py | A02·A03·A04·A05·A09·A11·A12·A13·A16·A17·A18·A19 + 구절 원문 일치(G1) | check.md, violations.csv |
| ④ 통계 | stats.py | 칸 채움 비율, 꼬리표 수렴 곡선, 근거 등급, 역량 질문 자동 점검 | stats.md, tag_growth.csv |
| ⑤ 답변 | answer.py | Holdout 질문에 B0(소개만)·B1(원문 발췌)·O1(관계도 근거)로 트윈 답 생성 | answers.jsonl, prompts_sample.md |
| ⑥ 비교 | compare.py | 실제 답과 트윈 답을 같은 추출기로 분해해 찬반·특징·가치·일화를 맞댐, Retest 천장 | compare.md |
| 합치기 | merge.py | 꼬리표를 뜻이 같은 것끼리 묶어 코드북 제안 → 사람이 고쳐 적용 | merge_proposal.csv |

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


## 설계표 v0.4 반영 (2026-11-23)

| 파일 | 바뀐 것 |
|---|---|
| common.py | 추출 프롬프트 v2.1: 결과(C) 꼬리표에 Schwartz 가치의 이름이나 뜻을 그대로 쓰지 않고 value로 보냄 (A19) |
| merge.py | C 칸 합치기 설명에 같은 규칙 |
| check.py | A13을 특징·결과에만 적용(가치는 Schwartz 칸이 처음부터 클래스), A19 검사 추가(C 꼬리표에 가치 이름이 들어가면 보고) |
| compare.py | "결과 겹침"(C 꼬리표 자카드) 지표 추가 |

graph.json에서 상자의 `hasValue`는 지금 본인 말(정리된 말)이고 Schwartz 칸은 `schwartz`에 있습니다. 설계표 v0.4 기준으로는 Schwartz 칸이 hasValue, 본인 말은 R24(phrase/valueWords)에 해당합니다. OWL로 옮길 때 이 대응으로 바꿉니다.
