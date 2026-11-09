import os, json, glob, unicodedata
root=os.path.expanduser("~/mnt"); OUT=[os.path.join(root,d) for d in os.listdir(root) if unicodedata.normalize("NFC",d)=="최종데이터"][0]
R={}
for f in glob.glob("restore/*.json"): R.update(json.load(open(f,encoding="utf-8")))
n=0
for p in glob.glob(f"{OUT}/[[]QA통합]*.json")+glob.glob(f"{OUT}/Train/[[]QA통합]*.json")+glob.glob(f"{OUT}/Holdout/[[]QA통합]*.json"):
    d=json.load(open(p,encoding="utf-8"))
    for it in d["items"]:
        if it["id"] in R:
            a,note=R[it["id"]]; it["answer"]=a; it["restored"]=True; it["restore_note"]=note; n+=1
    d["n_restored"]=sum(it["restored"] for it in d["items"])
    json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("applied",n,"of",len(R),"restorations; files:",len(glob.glob(f"{OUT}/[[]QA통합]*.json")+glob.glob(f"{OUT}/Train/[[]QA통합]*.json")+glob.glob(f"{OUT}/Holdout/[[]QA통합]*.json")))
