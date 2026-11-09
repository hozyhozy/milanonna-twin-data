# milanonna-twin-data

유튜버 **밀라논나(장명숙)** 의 공개 인터뷰·영상 발화로 디지털 트윈(페르소나 LLM)을 구축·검증하기 위한 데이터 저장소.
마케팅 인텔리전스 수업(2026-2, 한양대) 연구 과제의 작업 기록이며, 비공개(private) 저장소다.

## 데이터 개요 (2026-11-09 기준)

| 항목 | 값 |
|---|---|
| 영상 | 37편 (QA가 뽑힌 영상 36편; `9_tNDJQefbo`는 클립 중복으로 QA 0) |
| QA 문항 | **665** = Train 381 / Holdout 270 / Retest 14 |
| 엄격 Holdout (2025-01-01 이후) | 139 (Holdout의 부분집합, 강건성 검사용) |
| 문항 유형 | interview 276 (실제 질문) / derived 389 (발언에서 도출한 질문) |
| 자막 복원 완료 | 24문항 (`Hg_Zu2eSuOo` 13, `6_ktPoOUe-A` 11) |

### 분할 규칙 요약
1. **시간 기준 1차 분할** — 복귀(2024-08, 시즌 2) 이후 발화 = Holdout 후보, 이전 = Train (`split_time`).
2. **Holdout 후보를 Train 내용과 대조해 재배정** (`scripts/reassign.py`)
   - Train에 **같은 질문**이 있음 → `Retest` (본인 test-retest 일관성 = human ceiling 계산 전용)
   - 답에 Train에 이미 있는 **인생 일화**가 포함 → `Train`으로 이동
   - 둘 다 아님 → `Holdout` ("Train이 모르는 내용"만 남음)
3. `split_strict` = 2025-01-01 이후 기준. 누출(memorization) 반박용.

자세한 기준·판단 근거·필드 설명은 [`docs/분할기준_및_통합기록.md`](docs/분할기준_및_통합기록.md), 영상별 시기 판정은 [`docs/영상별_분할표.csv`](docs/영상별_분할표.csv).

## 폴더 구조

```
data/
  qa_merged/            # 통합 QA (분석에는 이 파일을 사용)
    전체.json           # 665문항, split 필드로 Train/Holdout/Retest 구분
    Train.json  Holdout.json  Retest.json
  qa_by_video/          # 영상별 QA 원본 (폴더 위치는 split_time 기준; 문항 단위 최종 분할은 통합 json의 split을 따름)
    Train/  Holdout/
docs/
  분할기준_및_통합기록.md     # 분할 규칙, 통합·중복 처리, 필드 설명, 복원 원칙, 재생성 방법
  영상별_분할표.csv           # 영상별 시기·근거·확신도·분할
  DSR_연구설계_초안_*.md       # 연구 설계 초안 (델타 기반 페르소나 LLM 충실도 프레임워크)
  참고자료/
    페르소나_사실_참고자료.md  # 배경 사실 (QA 학습 데이터 아님)
    [최종]언론기사_모음.md     # 기사별 직접 인용 1개 + 요약 — 배경 지식용, QA에서는 제외
scripts/
  merge_split.py        # 영상별 QA → 분할표·통합 json 생성 (DATES 수정 후 재실행)
  apply_restore.py      # restore/*.json 복원본을 통합 json에 반영
  reassign.py           # Holdout 후보를 Train 겹침 기준으로 Train/Holdout/Retest 재배정 (항상 마지막)
  dedup.py, dump.py, build2.py, run_specs.py, specs_*.py, specs/, restore/   # QA 추출·복원 보조
  youtube_subtitles_to_markdown.ipynb   # 유튜브 자막(.vtt) → 스크립트 md
WORKLOG.md              # 작업 일지
```

재생성 순서: `merge_split.py` → `apply_restore.py` → `reassign.py` (`docs/분할기준_및_통합기록.md` §6).
스크립트는 로컬 폴더(`~/Desktop/밀라논나/최종데이터`) 구조를 전제로 하므로 경로를 환경에 맞게 수정해야 한다.

## QA 항목 주요 필드

| 필드 | 뜻 |
|---|---|
| `id`, `video_id`, `title`, `channel`, `year`, `source_url` | 출처 (`id` = `영상ID_q###`) |
| `split` / `split_time` / `split_strict` | 최종 분할 / 시간 기준 분할 / 엄격 기준 분할 |
| `overlap`, `overlap_detail` | 시간상 Holdout 후보의 Train 겹침 판정 (same_question / shared_anecdote / novel) |
| `type`, `question`, `question_original` | interview·derived, 정리한 질문, 실제 질문 원문 |
| `topic`, `start`, `end` | 주제, 영상 내 구간(mm:ss) |
| `answer_raw` / `answer` | 자동자막 원문(수정 금지) / 복원본 |
| `restored`, `restore_note` | 복원 여부·근거 |
| `voice` | 본인 발화 / 2차 서술 구분 |
| `dup_group` | 반복 질문 묶음 (G01~G08) |

## 저장소에 올리지 않는 것 (로컬 보관)

- 유튜브 자막 전문: `youtube_transcripts/*.vtt`, `*.md`, `최종데이터/*/[최종]*.md`
- 이전 버전 통합 파일: `최종데이터/_old/`
- 내보내기용 tar: `_export_for_git.tar.gz`, `_repo_export_*.tar.gz`

QA의 `answer_raw`/`answer`는 발화 구간 발췌이므로 저장소에 포함하되, 영상 전체 전문은 올리지 않는다.

## 평가 시 유의

- Retest 14문항은 Train과 같은 질문이므로 **트윈 평가에서 제외**하고 human ceiling 계산에만 쓴다.
- 같은 일화가 여러 영상에 반복되므로 Holdout 평가는 **일화 회상형 / 의견·가치관형**으로 나눠 보고하는 것을 권장.
- 밀라논나는 공인이라 빈 LLM(G0)이 이미 알고 있을 수 있다 → 트윈 기여는 항상 **트윈 − G0** 로만 주장.
