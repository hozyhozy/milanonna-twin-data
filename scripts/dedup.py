# YouTube 자동자막(.vtt)의 "롤링 자막" 중복을 제거하는 스크립트
import re, sys, json, html           # re: 정규식, sys: 인자, json: 결과 저장, html: &gt; 같은 HTML 기호 복원

vtt = open(sys.argv[1], encoding="utf-8").read()   # 1번 인자로 받은 vtt 파일을 읽음
blocks = vtt.split("\n\n")                          # 빈 줄 기준으로 큐(cue) 단위로 나눔
time_re = re.compile(r"^(\d\d):(\d\d):(\d\d)\.\d+ -->")  # 큐 시작 시각을 찾는 패턴

lines_out = []      # (시작초, 새로 등장한 자막 한 줄) 목록
prev = None         # 직전에 저장한 줄 (같은 줄이 반복되면 건너뛰기 위함)
for b in blocks:
    rows = b.strip("\n").split("\n")            # 큐 안의 줄들
    m = time_re.match(rows[0]) if rows else None
    if not m:                                   # 시각 줄이 없으면 헤더 등이므로 건너뜀
        continue
    sec = int(m[1])*3600 + int(m[2])*60 + int(m[3])   # 시작 시각을 초로 변환
    text_rows = [r for r in rows[1:] if r.strip()]    # 공백뿐인 줄 제거
    if not text_rows:
        continue
    last = html.unescape(re.sub(r"<[^>]+>", "", text_rows[-1])).strip()  # 마지막 줄 = 새 자막, 단어별 타이밍 태그 제거
    if last and last != prev:                   # 직전 줄과 다를 때만 저장 → 중복 제거
        lines_out.append((sec, last))
        prev = last

json.dump(lines_out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)  # 결과 저장
