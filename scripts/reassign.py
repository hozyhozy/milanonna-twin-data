# Holdout 문항을 Train 내용과의 겹침에 따라 Train / Holdout / Retest 로 재배정
import os, re, json, itertools, unicodedata, glob
from collections import Counter, defaultdict
root=os.path.expanduser("~/mnt"); OUT=[os.path.join(root,d) for d in os.listdir(root) if unicodedata.normalize("NFC",d)=="최종데이터"][0]
ANEC = {  # 일화 코드: 정규식 (답 본문에서 검색)
 "삼풍붕괴":   r"삼풍|백화점이 무너|백화점 붕괴|무너졌|목요일에 사고",
 "86아시안":   r"아시안\s?게임|86년.*의상|개[·.]?폐[회막]식",
 "큰아들수술": r"큰아들.{0,15}(수술|병원|생사)|(수술|뇌).{0,10}큰아들|고3.{0,20}수술",
 "흰머리염색": r"염색|흰머리|백발|머리가 새하얗",
 "78유학":     r"78년|1978|마랑고니|(최초|처음).{0,15}유학|유학.{0,10}(갔|떠났|길)",
 "돌체동기":   r"돌체|도메니코",
 "명품유치":   r"페라가모|막스\s?마라|명품관|명품.{0,20}(들여|론칭|수입)",
 "유튜브시작": r"(유튜브|채널).{0,30}(권유|캐스팅|후배|빙수|빵집)|길거리 캐스팅|68세에|유튜브를 시작하게|유튜브를 하게 된",
 "밀라논나뜻": r"논나.{0,20}(뜻|의미|이탈리아 말|이태리 말|이탈리아어)|밀라노 할머니",
 "시신기증":   r"시신|장기\s?기증|기증",
 "수익기부":   r"기부|인세|보육원|수익.{0,10}(다|전액)",
 "어머니별세": r"(친정)?어머니.{0,20}(돌아가|별세|장례)",
 "활동중단복귀":r"(2년|이년).{0,20}(쉬|중단|복귀)|시즌\s?2|번아웃",
 "아들런던":   r"런던|둘째.{0,15}(미술|작가|아티스트|전시)",
 "명예기사":   r"기사\s?작위|명예\s?기사",
 "100년셔츠":  r"100년.{0,10}(옷|셔츠)|할머니가 만드신",
 "보자기재활용":r"보자기",
 "스칼라좌":   r"스칼라|라 스칼라",
 "꽃자리기도": r"꽃자리|앉은 자리",
 "공주피난출생":r"공주|피난.{0,20}(태어|낳)",
 "병아리다리": r"병아리",
 "건빵점심":   r"건빵",
 "36시간비행": r"36시간|서울서 대만|방콕",
}
norm=lambda s: re.sub(r"[^\w가-힣]","",s)
TEMPLATE=r"에 대해|어떻게|무엇|뭐라고|생각하나요|생각하시나요|하나요|인가요|있나요|했나요|됐나요|나요|까요|이란|라고|이라고|은 |는 |이 |가 |을 |를 |의 |에서|으로|로 |때문|이유|계기|무슨|어떤|가장|요즘|지금|한마디로|마지막으로|해 주고 싶은|해 주세요"
def qgrams(s):
    s=norm(re.sub(TEMPLATE," ",s)); return set(s[i:i+2] for i in range(len(s)-1))
def anecs(text): return {k for k,p in ANEC.items() if re.search(p,text)}

items=json.load(open(f"{OUT}/[QA통합]전체.json",encoding="utf-8"))["items"]
train=[i for i in items if i["split"]=="Train"]; hold=[i for i in items if i["split"]=="Holdout"]
train_anecs=Counter(); [train_anecs.update(anecs(i["answer_raw"])) for i in train]
train_q=[(qgrams(i["question"]),i) for i in train]
report=defaultdict(list)
for h in hold:
    # 1) 같은 질문: dup_group 공유 또는 질문 유사도 ≥0.6
    same=None
    if h["dup_group"] and any(t["dup_group"]==h["dup_group"] for t in train): same=h["dup_group"]
    else:
        g=qgrams(h["question"])
        for tg,t in train_q:
            if len(g)>=3 and len(g&tg)/max(1,len(g|tg))>=0.5: same=t["question"]; break
    a=anecs(h["answer_raw"]); shared=sorted(k for k in a if train_anecs[k]>0)
    if same: h["split_final"]="Retest"; h["overlap"]="same_question"; h["overlap_detail"]=same
    elif shared: h["split_final"]="Train"; h["overlap"]="shared_anecdote"; h["overlap_detail"]=",".join(shared)
    else: h["split_final"]="Holdout"; h["overlap"]="novel"; h["overlap_detail"]=None
    report[h["overlap"]].append(h)
for t in train: t["split_final"]="Train"; t["overlap"]=None; t["overlap_detail"]=None
for i in items: i["split_time"]=i["split"]; i["split"]=i["split_final"]; del i["split_final"]
print({k:len(v) for k,v in report.items()})
print("\n[Retest 전체]"); [print(f"  {h['video_id']} | {h['question']}  ← {h['overlap_detail']}") for h in report["same_question"]]
print("\n[shared_anecdote 일화별]", Counter(k for h in report["shared_anecdote"] for k in h["overlap_detail"].split(",")))
print("\n[shared_anecdote 전체]"); [print(f"  {h['video_id']} | {h['question']}  ← {h['overlap_detail']}") for h in report["shared_anecdote"]]
print("\n[novel 예시 10]"); [print(f"  {h['video_id']} | {h['question']}") for h in report["novel"][:10]]
meta={"note":"split=최종 분할(Train/Holdout/Retest). split_time=시간 기준(복귀 후=Holdout) 분할. split_strict=2025년 이후 기준. overlap: 시간상 Holdout이던 문항이 Train 내용과 겹치는지 — same_question(같은 질문이 Train에 있음 → Retest: 본인 일관성 천장 계산 전용), shared_anecdote(Train에 있는 인생 일화를 답에 포함 → Train으로 이동), novel(겹침 없음 → Holdout 유지). 일화 사전은 reassign.py의 ANEC 참조.",
      "split_rule":"Holdout = 복귀(2024-08) 이후 발화 중 Train과 같은 질문도, 겹치는 일화도 없는 문항. Retest = 복귀 후 발화 중 Train에 같은 질문이 있는 문항. 그 외 = Train."}
def dump(path, its): json.dump({**meta,"n_items":len(its),"items":its},open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
dump(f"{OUT}/[QA통합]전체.json",items)
for sp in ("Train","Holdout","Retest"):
    os.makedirs(f"{OUT}/{sp}",exist_ok=True); its=[i for i in items if i["split"]==sp]; dump(f"{OUT}/{sp}/[QA통합]{sp}.json",its); print(sp,len(its))
