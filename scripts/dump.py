# 영상별로 중복 제거 + 읽기용 텍스트(1분 단위 묶음) 출력
import json, os, sys, glob, subprocess, unicodedata
W = os.path.expanduser("~/work")
SRC = os.path.expanduser("~/mnt/youtube_transcripts")
for f in sorted(glob.glob(f"{SRC}/*.vtt")):
    vid = f[:-7][-11:]                                  # 파일명 끝 11글자 = 유튜브 영상 ID
    out = f"{W}/lines/{vid}.json"
    subprocess.run(["python3", f"{W}/dedup.py", f, out], check=True)   # 앞에서 만든 중복 제거 스크립트 재사용
    if len(sys.argv) > 1 and vid not in sys.argv[1:]:
        continue
    L = json.load(open(out, encoding="utf-8"))
    print("=====", vid, os.path.basename(f)[:60])
    block, bstart = [], None
    for sec, t in L:
        if bstart is None: bstart = sec
        block.append(t.replace(">>", "⟫").replace("[음악]", "").strip())
        if sec - bstart >= 45:                          # 약 45초마다 한 단락
            print(f"[{bstart//60:02d}:{bstart%60:02d}] " + " ".join(block)); block, bstart = [], None
    if block: print(f"[{bstart//60:02d}:{bstart%60:02d}] " + " ".join(block))
