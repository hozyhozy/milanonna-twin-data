#!/usr/bin/env python3
"""온톨로지 파일럿 한 번에 돌리기 (저장소 루트에서 실행).

실행 환경: answer·compare 단계는 e5 임베딩을 쓰므로 반드시 .venv-st/bin/python 으로 실행합니다.
  (.venv 의 Python 3.14 에는 sentence-transformers가 없어 시작할 때 멈춥니다. 일부러 TF-IDF로 돌리려면
   --retrieval tfidf --match char 또는 EMBED_ALLOW_TFIDF=1)

.venv-st/bin/python experiments/ontology/run_pilot.py --run pilot4 --copy-extract-from pilot3 \
    --stages build,check,stats,answer,compare --n 50 --clean --reuse-truth-from pilot3
    # 추출(381회)은 pilot3 것을 그대로 쓰고, 관계도 → 답 생성 → 비교만 새로

python3 experiments/ontology/run_pilot.py --run pilot1 --limit 30 --n 20      # 작은 시험: Train 30발화, Holdout 20문항
python3 experiments/ontology/run_pilot.py --run pilot2 --limit 30 --n 20 --clean   # v2: 자막 교정본으로
python3 experiments/ontology/run_pilot.py --run pilot1 --stages extract,build,check,stats   # 관계도까지만
MOCK_LLM=1 python3 experiments/ontology/run_pilot.py --run mock                # API 없이 점검

단계: (clean) → extract → build → check → stats → answer → compare
꼬리표 합치기(merge.py)는 사람이 CSV를 고쳐야 해서 따로 실행합니다 (README 참고).
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--limit", type=int, default=30, help="Train 앞 N발화 (0=전체 381)")
    ap.add_argument("--n", type=int, default=20, help="Holdout 문항 수 (최대 50)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tags", default="")
    ap.add_argument("--stages", default="extract,build,check,stats,answer,compare")
    ap.add_argument("--clean", action="store_true", help="자막 교정(clean.py) 후 교정본으로 추출·비교")
    ap.add_argument("--no-ceiling", action="store_true", help="Retest 천장 계산 생략")
    ap.add_argument("--retrieval", default="emb", choices=["emb", "tfidf"], help="answer: B1·O1 근거 찾기")
    ap.add_argument("--match", default="emb", choices=["emb", "char"], help="compare: 대상·일화 맞추기")
    ap.add_argument("--copy-extract-from", default="", help="이 run의 extract.jsonl을 복사해 추출을 건너뜀")
    ap.add_argument("--reuse-truth-from", default="", help="compare: 정답 추출·천장 행을 이 run에서 가져옴")
    a = ap.parse_args()
    stages = a.stages.split(",")
    if ({"answer", "compare"} & set(stages)) and (a.retrieval == "emb" or a.match == "emb"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "scripts"))
        from embed import require_st
        require_st("run_pilot (answer·compare)")       # .venv-st가 아니면 여기서 멈춤
    runs = os.path.join(os.path.dirname(HERE), "runs", "ontology")
    if a.copy_extract_from:
        os.makedirs(os.path.join(runs, a.run), exist_ok=True)
        for fn in ("extract.jsonl", "tags.json"):          # 추출 결과와 그때의 꼬리표 목록
            src, dst = os.path.join(runs, a.copy_extract_from, fn), os.path.join(runs, a.run, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"추출 결과 복사: {src} → {dst}")
    c = ["--clean"] if a.clean else []
    cmds = {
        "clean": ["clean.py", "--split", "Train"] + (["--limit", str(a.limit)] if a.limit else []),
        "clean_h": ["clean.py", "--split", "Holdout", "--ids-from-holdout50"],
        "clean_r": ["clean.py", "--split", "Retest"],
        "extract": ["extract.py", "--run", a.run, "--limit", str(a.limit)] + (["--tags", a.tags] if a.tags else []) + c,
        "build": ["build.py", "--run", a.run],
        "check": ["check.py", "--run", a.run],
        "stats": ["stats.py", "--run", a.run],
        "answer": ["answer.py", "--run", a.run, "--n", str(a.n), "--k", str(a.k), "--retrieval", a.retrieval] + c,
        "compare": ["compare.py", "--run", a.run, "--match", a.match] + c + ([] if a.no_ceiling else ["--ceiling"])
                   + (["--reuse-truth-from", a.reuse_truth_from] if a.reuse_truth_from else []),
    }
    if a.clean and "clean" not in stages:
        stages = ["clean", "clean_h", "clean_r"] + stages
    for s in stages:
        print(f"\n===== {s} =====")
        r = subprocess.run([sys.executable, os.path.join(HERE, cmds[s][0])] + cmds[s][1:], cwd=HERE)
        if r.returncode:
            raise SystemExit(f"{s} 단계 실패")
    print(f"\n결과: experiments/runs/ontology/{a.run}/ (stats.md, check.md, compare.md, sheet_*.csv)")


if __name__ == "__main__":
    main()
