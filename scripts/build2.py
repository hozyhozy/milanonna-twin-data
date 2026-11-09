# 영상 1개 처리: 중복 제거된 자막 + 영상별 설정(spec) → [최종] 스크립트 md, [QA] 질문-답 json
# 사용법: python3 build2.py specs/<영상ID>.json
import json, os, re, sys, unicodedata                 # 파일 입출력, 정규식, 한글 파일명 정규화

W = os.path.expanduser("~/work")
MNT = os.path.expanduser("~/mnt")
def folder(name):                                      # macOS 한글 폴더명(NFD)도 찾을 수 있게 정규화해서 비교
    for d in os.listdir(MNT):
        if unicodedata.normalize("NFC", d) == name: return os.path.join(MNT, d)
    raise SystemExit(f"폴더 없음: {name}")
OUT = folder("최종데이터")                              # 결과 저장 폴더

spec = json.load(open(sys.argv[1], encoding="utf-8")) # 영상별 설정 읽기
vid = spec["video_id"]
lines = json.load(open(f"{W}/lines/{vid}.json", encoding="utf-8"))  # [(초, 자막줄)] — dedup.py 결과

def clean(t):                                          # 자막 조각 정리
    t = t.replace(">>", " ").replace("[음악]", " ")      # 화자 표시(>>)와 배경음악 표시 제거 (화자는 spec에서 지정)
    return re.sub(r"\s+", " ", t).strip()              # 공백 정리

# 1) 영상 전체를 하나의 긴 글로 이어 붙이고, 글자 위치마다 시각을 기억
text, offs = "", []                                    # offs = [(글자 위치, 초)]
for sec, t in lines:
    c = clean(t)
    if not c: continue
    if text: text += " "
    offs.append((len(text), sec)); text += c
def sec_at(pos):                                       # 글자 위치 → 그 글자가 나온 시각
    s = offs[0][1]
    for o, sec in offs:
        if o <= pos: s = sec
        else: break
    return s
def sec_after(pos):                                    # 글자 위치 다음 조각의 시작 시각(구간 끝 시각으로 사용)
    for o, sec in offs:
        if o > pos: return sec
    return spec["duration_sec"]

# 2) 화자 경계: spec["bounds"] = [[경계 문구, 화자코드], ...] (앞에서부터 순서대로 찾음)
cuts, cur = [(0, spec["first"])], 0
for marker, spk in spec.get("bounds", []):
    p = text.find(marker, cur)                         # 직전 경계 이후에서 문구 찾기
    assert p >= 0, ("경계 문구 없음", marker)
    cuts.append((p, spk)); cur = p + 1
segs = []                                              # 발화 목록 [시작위치, 끝위치, 화자]
for k, (p, spk) in enumerate(cuts):
    q = cuts[k+1][0] if k+1 < len(cuts) else len(text)
    if text[p:q].strip():
        if segs and segs[-1][2] == spk: segs[-1][1] = q   # 같은 화자가 이어지면 합침
        else: segs.append([p, q, spk])

def mmss(x): return f"{x//60:02d}:{x%60:02d}"
NAME = spec["speakers"]                                # 화자코드 → 표시 이름

# 3) [최종] 스크립트 md
title, base = spec["title"], spec["base"]
md = [f"# [최종] {title}", "",
      f"- 채널: {spec['channel']}", f"- 업로드: {spec['upload_date']}",
      f"- URL: https://www.youtube.com/watch?v={vid}",
      f"- 원본 자막: `{base}.vtt` / 원본 md: `{base}.md`",
      "- 정리 방식: 자동자막의 롤링 중복 제거, `[음악]` 표시 삭제, 화자 구분은 내용을 보고 수동 지정",
      "- 주의: 자동자막 원문의 오탈자는 음성 확인 없이 고치지 않고 그대로 둠"]
if spec.get("note"): md.append(f"- 참고: {spec['note']}")
md += ["", "---", ""]
for p, q, spk in segs:
    md += [f"**[{mmss(sec_at(p))}] {NAME[spk]}:** {text[p:q].strip()}", ""]
def fname(prefix, ext):                               # "[최종]제목 - 파일이름" 형식, 255바이트를 넘으면 "[최종]파일이름"으로 대체
    n = f"{prefix}{title.replace('/', '⧸')} - {base}{ext}"
    return n if len(n.encode()) <= 250 else f"{prefix}{base}{ext}"
open(f"{OUT}/{fname('[최종]', '.md')}", "w", encoding="utf-8").write("\n".join(md))

# 4) [QA] 질문-답: spec["qa"] = [[유형, 질문, 답 시작 문구, 답 끝 문구, 주제, 원래 질문 찾는 문구|null], ...]
target = spec.get("answer_speaker", "M")               # 답이 될 화자(기본: 밀라논나)
def owner(pos):                                        # 글자 위치가 속한 발화의 화자
    for p, q, spk in segs:
        if p <= pos < q: return spk
items, used = [], []
for n, (typ, qq, a0, a1, topic, qm) in enumerate(spec["qa"], 1):
    assert text.count(a0) == 1, ("시작 문구가 유일하지 않음", a0, text.count(a0))
    i0 = text.index(a0)
    i1 = text.index(a1, i0) + len(a1)                  # 끝 문구까지 포함
    assert owner(i0) == target and owner(i1 - 1) == target, ("답 구간이 밀라논나 발화가 아님", qq)
    for u0, u1 in used:                                # 답 구간끼리 겹치면 중단(중복 방지)
        assert not (i0 < u1 and u0 < i1), ("구간 겹침", qq)
    used.append((i0, i1))
    it = {"id": f"q{n:03d}", "type": typ, "question": qq, "answer": text[i0:i1],
          "start": mmss(sec_at(i0)), "end": mmss(sec_after(i1 - 1)), "topic": topic}
    if qm:                                             # 실제 질문(진행자·제작진·자막)의 원문을 함께 기록
        j = text.find(qm); assert j >= 0, ("원래 질문 문구 없음", qm)
        seg = next(s for s in segs if s[0] <= j < s[1])
        it["question_original"] = text[seg[0]:seg[1]].strip() if owner(j) != target else qm
    items.append(it)
assert len(set(i["question"] for i in items)) == len(items), "질문 중복"
out = {"video_id": vid, "title": title, "channel": spec["channel"], "upload_date": spec["upload_date"],
       "source_file": f"{base}.md",
       "note": "answer는 자동자막 원문 그대로(오탈자 미수정). type=interview는 진행자·제작진이 실제로 한 질문(또는 영상 속 질문 자막·사연), derived는 발언 내용에서 도출한 질문. 답 구간은 서로 겹치지 않음." + (" " + spec["note"] if spec.get("note") else ""),
       "n_items": len(items), "items": items}
json.dump(out, open(f"{OUT}/{fname('[QA]', '.json')}", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(vid, "segments:", len(segs), "qa:", len(items))
