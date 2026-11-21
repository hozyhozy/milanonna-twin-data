#!/usr/bin/env python3
"""온톨로지 파일럿 한 번에 돌리기 (저장소 루트에서 실행).

python3 experiments/ontology/run_pilot.py --run pilot1 --limit 30 --n 20      # 작은 시험: Train 30발화, Holdout 20문항
python3 experiments/ontology/run_pilot.py --run pilot1 --stages extract,build,check,stats   # 관계도까지만
MOCK_LLM=1 python3 experiments/ontology/run_pilot.py --run mock                # API 없이 점검

단계: extract → build → check → stats → answer → compare
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--limit", type=int, default=30, help="Train 앞 N발화 (0=전체 381)")
    ap.add_argument("--n", type=int, default=20, help="Holdout 문항 수 (최대 50)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tags", default="")
    ap.add_argument("--stages", default="extract,build,check,stats,answer,compare")
    a = ap.parse_args()
    cmds = {
        "extract": ["extract.py", "--run", a.run, "--limit", str(a.limit)] + (["--tags", a.tags] if a.tags else []),
        "build": ["build.py", "--run", a.run],
        "check": ["check.py", "--run", a.run],
        "stats": ["stats.py", "--run", a.run],
        "answer": ["answer.py", "--run", a.run, "--n", str(a.n), "--k", str(a.k)],
        "compare": ["compare.py", "--run", a.run],
    }
    for s in a.stages.split(","):
        print(f"\n===== {s} =====")
        r = subprocess.run([sys.executable, os.path.join(HERE, cmds[s][0])] + cmds[s][1:], cwd=HERE)
        if r.returncode:
            raise SystemExit(f"{s} 단계 실패")
    print(f"\n결과: experiments/runs/ontology/{a.run}/ (stats.md, check.md, compare.md, sheet_*.csv)")


if __name__ == "__main__":
    main()
