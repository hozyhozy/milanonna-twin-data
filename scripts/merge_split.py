import os, re, json, csv, shutil, unicodedata, glob
root=os.path.expanduser("~/mnt"); OUT=[os.path.join(root,d) for d in os.listdir(root) if unicodedata.normalize("NFC",d)=="최종데이터"][0]
# 영상별 시기 판정: (연도 추정, 근거, 확신도)  — 기준: 2025-01-01 이후 = Holdout
DATES = {
 "5rEd9woe2rE":("2025-12-29","제목/업로드일","확인"),
 "p4kvkbIXOgU":("2026-04-01","업로드일","확인"),
 "M_l5PA5Nhjk":("2025-04-28","업로드일","확인"),
 "hLLWZ6VjroI":("2025-06-20","KBS 사사건건 방송일","확인"),
 "iOh-N63_aUM":("2025","서장훈의 이웃집 백만장자 7회(2025년 방송)","확인"),
 "44pG56vzkY4":("2025","서장훈의 이웃집 백만장자 7회(2025년 방송)","확인"),
 "Z6FzvfTxrLc":("2024(추정)","2022.7 중단 후 약 2년 뒤 복귀 영상","추정"),
 "gPIgX2rsMQ4":("2024(추정)","두 번째 책 『오롯이 내 인생이잖아요』(2024.8) 출간 북토크","추정"),
 "O7nTrK3lXaI":("2024(추정)","제작자 이경신과 공동 출연 → 공저 출간(2024.8) 전후","추정"),
 "WumK6R8ER-4":("2024~(추정)","복귀 후 형식(제목에 [eng] 없음, 사연 상담 시즌2)","추정"),
 "GqImaMrZDb8":("2024~(추정)","복귀 후 형식(제목에 [eng] 없음)","추정"),
 "5D8jTU3js4U":("2024~(추정)","복귀 후 형식(제목에 [eng] 없음)","추정"),
 "u9z7_SjN28U":("2024~(추정)","복귀 후 형식(제목에 [eng] 없음)","추정"),
 "A1YLaUFL3IY":("2024~(추정)","복귀 후 형식(제목에 [eng] 없음)","추정"),
 "TwatJuhJlPQ":("2020-05-15","업로드일","확인"),
 "fL7O8M4q__c":("2020","유퀴즈 시즌3 51회(2020년 4월)","확인"),
 "Bft2CIRT6-o":("2021-07-15","KBS 대화의 희열 방송일","확인"),
 "9_tNDJQefbo":("2021-07-15","KBS 대화의 희열(클립)","확인"),
 "E1KtGlptkdE":("2021(추정)","첫 책 『햇빛은 찬란하고…』(2021.8) 출간 인터뷰","추정"),
 "oiOdmNkDCUM":("2021(추정)","첫 책 북토크","추정"),
 "ksZWwsEOi6Y":("2021(추정)","첫 책 출간 인터뷰(엘르)","추정"),
 "1-HvAWSw51c":("2020(추정)","구독자 20만 기념 → 2020년 상반기","추정"),
 "42X62gp55LI":("2020~21(추정)","[eng] 논나의 아지트 시리즈","추정"),
 "GUwGWDuBSJs":("2020~21(추정)","[eng] 논나의 아지트 시리즈","추정"),
 "mNBrRKYBhE0":("2020~21(추정)","[eng] 논나의 아지트 시리즈","추정"),
 "n10NtX_VBBA":("2020~21(추정)","논나의 아지트 시리즈","추정"),
 "6_ktPoOUe-A":("2020~21(추정)","[eng] 초기 룸투어","추정"),
 "Kr3zgm9-K6U":("2022(추정)","영상 속 '결혼 47년','85년 구입 37년' 발언","추정"),
 "Hg_Zu2eSuOo":("2021(추정)","영상 속 첫 책 홍보 + 94년 구입 옷이 27년 됐다는 발언","추정"),
 "VNPdom_zir0":("2021~22(추정)","EBS 초대석 — 활동 중단(2022.7) 이전 방송으로 추정","추정·확인 필요"),
 "tgQhTZNeEHo":("2023(약)","유튜브 표시 '3년 전'(2026.11 기준) — 사용자 확인","확인"),
 "NPYE_-uKpDU":("2021(약)","유튜브 표시 '5년 전'(2026.11 기준) — 사용자 확인","확인"),
 "fwbYEKXB39A":("2021(약)","유튜브 표시 '5년 전'(2026.11 기준) — 사용자 확인","확인"),
 "5mMt8SToWX8":("2021(추정)","Draw My Life — 영상 속 '69년 인생' 발언(1952년생)","추정"),
 "pvKbmUTQHtg":("2020-03(추정)","영상 속 이탈리아 코로나 봉쇄 상황, '11월 입국·3월 초 출국' 언급","추정"),
 "OayzNAKDQiI":("2025-03(추정)","영상 속 '전시 오프닝 3월 22일 토요일'(2025년) + 사용자 확인 '1년 전'","확인"), "press_collection":("2020~2021","언론 기사 4건(기사별 날짜는 항목 id 참조)","확인"), "FNfQ6IQQjao":("2024-10-30","MBC 라디오스타 방송일","확인"),
}
def split_strict(vid):                       # 엄격 기준: 2025-01-01 이후 = Holdout
    y=DATES[vid][0]
    return "Holdout" if y.startswith("2025") or y.startswith("2026") else "Train"
def split_of(vid):                           # 주 기준: 복귀(2024-08) 이후 = Holdout (채널 공백 2022.7~2024.8 → 2024년 자료는 전부 복귀 후)
    y=DATES[vid][0]
    return "Holdout" if y.startswith("2024") or split_strict(vid)=="Holdout" else "Train"
# 같은 질문을 여러 영상에서 한 경우: 하나의 질문군(dup_group)으로 묶음 (답은 서로 다른 발화라 모두 보존)
GROUPS = {
 "G01 밀라논나 뜻": ["'밀라논나'는 무슨 뜻인가요?"],
 "G02 유튜브 시작 계기": ["유튜브는 왜 시작하게 되었나요?","유튜브는 어떻게 시작하게 되었나요?","유튜브를 시작하게 된 계기는?","유튜브를 하게 된 계기는?"],
 "G03 롤모델이라는 말": ["롤모델이라는 말을 들으면 어떤가요?"],
 "G04 부모님과의 이별 준비": ["부모님과의 이별은 어떻게 준비해야 할까요?"],
 "G05 인생이란": ["인생이란 무엇이라고 생각하나요?"],
 "G06 용기와 두려움": ["용기와 두려움은 어디에서 온다고 생각하나요?"],
 "G07 흰머리 염색 중단": ["흰머리 염색을 그만둔 이유는 무엇인가요?","흰머리 염색을 그만둔 이유는?"],
 "G08 하고 싶은 것이 있을 때": ["하고 싶은 것이 있을 때 어떻게 하라고 하나요?","해 보고 싶은 것이 있을 때 어떻게 하라고 조언하나요?"],
}
Q2G={q:g for g,qs in GROUPS.items() for q in qs}
files=sorted(p for sub in ("","Train","Holdout") for p in glob.glob(os.path.join(OUT,sub,"[[]QA]*.json")))
items=[]; manifest=[]
for f in files:
    d=json.load(open(f,encoding="utf-8")); vid=d["video_id"]
    y,basis,conf=DATES[vid]; sp=split_of(vid)
    manifest.append(dict(video_id=vid,title=d["title"],channel=d["channel"],year=y,basis=basis,confidence=conf,split=sp,split_strict=split_strict(vid),n_qa=d["n_items"]))
    for it in d["items"]:
        items.append({"id":f"{vid}_{it['id']}","video_id":vid,"title":d["title"],"channel":d["channel"],"year":y,"split":sp,
            "split_strict":split_strict(vid),"type":it["type"],"question":it["question"],"question_original":it.get("question_original"),
            "topic":it["topic"],"start":it["start"],"end":it["end"],
            "answer_raw":it["answer"],"answer":it["answer"],"restored":False,"restore_note":None,"voice":it.get("voice","verbatim_asr"),"source_url":it.get("source_url"),
            "dup_group":Q2G.get(it["question"])})
# 중복 제거: 질문+답 완전 동일 항목(영상 간 클립 중복 등)
seen=set(); uniq=[]
for it in items:
    k=(re.sub(r"\s","",it["question"]),re.sub(r"\s","",it["answer_raw"]))
    if k in seen: continue
    seen.add(k); uniq.append(it)
print("items",len(items),"after exact dedup",len(uniq))
# 질문군이 Train/Holdout에 걸치는지 표시(누수 점검용)
for g in GROUPS:
    for col in ("split","split_strict"):
        sps={it[col] for it in uniq if it["dup_group"]==g}
        if len(sps)>1: print(f"누수 주의[{col}]:",g)
meta={"note":"answer_raw=자동자막 원문. answer=복원본(restored=true일 때만 다름; 복원은 발음 오인식만 고치고 뜻이 불명한 부분은 [불명]으로 표시). dup_group=여러 영상에서 반복된 같은 질문 묶음(답은 각기 다른 발화라 모두 보존). split 기준: 2025-01-01 이후 업로드/방송 = Holdout.",
      "split_rule":"Holdout = 2025년 이후(LLM 학습 데이터 이후 가능성이 높은 최신 발화), Train = 그 이전"}
def dump(path, its):
    json.dump({**meta,"n_items":len(its),"items":its},open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
dump(f"{OUT}/[QA통합]전체.json",uniq)
for sp in ("Train","Holdout"):
    os.makedirs(f"{OUT}/{sp}",exist_ok=True)
    its=[it for it in uniq if it["split"]==sp]; dump(f"{OUT}/{sp}/[QA통합]{sp}.json",its); print(sp,len(its))
with open(f"{OUT}/영상별_분할표.csv","w",encoding="utf-8-sig",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(sorted(manifest,key=lambda r:(r["split"],r["year"])))
# 영상별 md/json 파일을 Train/Holdout 폴더로 이동
for f in os.listdir(OUT):
    g=unicodedata.normalize("NFC",f)
    if (g.startswith("[최종]") or g.startswith("[QA]")) and not g.startswith("[QA통합]"):
        vid=next((v for v in DATES if v in f), "press_collection" if "언론기사" in g else None)
        if vid: shutil.move(os.path.join(OUT,f), os.path.join(OUT,split_of(vid),f))
        else: print("분할 미지정 파일(루트에 둠):", f)
for sp in ("Train","Holdout"): print(sp, len(os.listdir(f"{OUT}/{sp}")),"files")
