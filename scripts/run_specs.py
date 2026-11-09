# specs_*.py의 설정과 spec 뼈대를 합쳐 json으로 저장한 뒤 build2.py 실행
import json, sys, subprocess, importlib
mod = importlib.import_module(sys.argv[1])
WRONG = "남의 말에 휘둘리지 않는 법 #밀라논나"           # 원본 md 제목이 잘못 들어간 경우
for vid, sp in mod.S.items():
    sk = json.load(open(f"specs/{vid}.skel.json", encoding="utf-8"))
    if sk["title"] == WRONG: sk["title"] = sk["base"][:-15]   # 파일명에서 제목 복원('_영상ID.ko' 제거)
    sk.update(sp); sk["qa"] = [list(x) for x in sp["qa"]]
    json.dump(sk, open(f"specs/{vid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    r = subprocess.run(["python3", "build2.py", f"specs/{vid}.json"], capture_output=True, text=True)
    print(r.stdout.strip() or (vid + " ERROR " + r.stderr.strip().splitlines()[-1]))
