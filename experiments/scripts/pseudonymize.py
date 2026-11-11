#!/usr/bin/env python3
"""가명 처리: 실명·고유명사를 치환 사전(experiments/config/pseudonym_dict.json)에 따라 바꾼 사본을 만든다.

입력 (원본은 건드리지 않음)
  data/qa_merged/{전체,Train,Holdout,Retest}.json
  docs/참고자료/페르소나_사실_참고자료.md
  docs/참고자료/[최종]언론기사_모음.md
출력
  data/anon/{전체,Train,Holdout,Retest}.json   (QA 필드만 치환, 메타 필드 유지)
  data/anon/페르소나_사실_참고자료.md, 언론기사_모음.md
  experiments/results/pseudonymize_report.md    (규칙별 치환 횟수 + 잔여 의심어 목록)

사용
  python experiments/scripts/pseudonymize.py            # 기본 경로
  python experiments/scripts/pseudonymize.py --dict path/to/dict.json
"""
import argparse, json, os, re, sys, collections, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_FIELDS = ["question", "question_original", "answer", "answer_raw", "title", "channel",
               "restore_note", "overlap_detail", "topic"]
QA_FILES = ["전체", "Train", "Holdout", "Retest"]
MD_FILES = {"docs/참고자료/페르소나_사실_참고자료.md": "페르소나_사실_참고자료.md",
            "docs/참고자료/[최종]언론기사_모음.md": "언론기사_모음.md"}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def load_dict(path):
    d = json.load(open(path, encoding="utf-8"))
    rules = [(nfc(a), b) for a, b in d["rules"]]
    # 사전에 적힌 순서대로 적용한다(앞 규칙 우선). 복합어를 앞에 둘 것.
    regex_rules = [(re.compile(p), r) for p, r in d.get("regex_rules", [])]
    return d, rules, regex_rules


class Replacer:
    def __init__(self, rules, regex_rules):
        self.rules, self.regex_rules = rules, regex_rules
        self.counts = collections.Counter()

    def __call__(self, text):
        if not text:
            return text
        text = nfc(text)
        for pat, rep in self.regex_rules:
            text, n = pat.subn(rep, text)
            self.counts["re:" + pat.pattern] += n
        for a, b in self.rules:
            n = text.count(a)
            if n:
                text = text.replace(a, b)
                self.counts[a] += n
        return text


def residual_scan(texts, terms):
    found = collections.Counter()
    examples = {}
    for t in texts:
        for term in terms:
            for m in re.finditer(re.escape(term), t):
                found[term] += 1
                examples.setdefault(term, t[max(0, m.start() - 25): m.end() + 25].replace("\n", " "))
    return found, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default=os.path.join(ROOT, "experiments/config/pseudonym_dict.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/anon"))
    args = ap.parse_args()

    d, rules, regex_rules = load_dict(args.dict)
    rep = Replacer(rules, regex_rules)
    os.makedirs(args.out, exist_ok=True)
    all_texts = []

    for name in QA_FILES:
        src = os.path.join(ROOT, "data/qa_merged", f"{name}.json")
        obj = json.load(open(src, encoding="utf-8"))
        for it in obj["items"]:
            for f in TEXT_FIELDS:
                if it.get(f):
                    it[f] = rep(it[f])
                    all_texts.append(it[f])
            it["persona_code"] = d["persona"]["code"]
            it.pop("source_url", None)  # 유튜브 URL은 영상ID로 재식별 가능 → 제거
        obj["note"] = "[가명 처리본] " + rep(obj.get("note", ""))
        obj["split_rule"] = rep(obj.get("split_rule", ""))
        json.dump(obj, open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"{name}: {len(obj['items'])} items → data/anon/{name}.json")

    for src, dst in MD_FILES.items():
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            continue
        txt = rep(open(p, encoding="utf-8").read())
        txt = re.sub(r"https?://\S+", "[URL 제거]", txt)
        all_texts.append(txt)
        open(os.path.join(args.out, dst), "w", encoding="utf-8").write(txt)
        print(f"{src} → data/anon/{dst}")

    found, examples = residual_scan(all_texts, d.get("residual_scan", []))
    os.makedirs(os.path.join(ROOT, "experiments/results"), exist_ok=True)
    rp = os.path.join(ROOT, "experiments/results/pseudonymize_report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("# 가명 처리 리포트\n\n## 규칙별 치환 횟수\n\n| 원문 | 횟수 |\n|---|---:|\n")
        for k, v in rep.counts.most_common():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## 잔여 의심어 (0이어야 함)\n\n")
        if not found:
            f.write("없음\n")
        for k, v in found.most_common():
            f.write(f"- **{k}** ×{v} — 예: …{examples[k]}…\n")
    print(f"\n치환 {sum(rep.counts.values())}회, 잔여 의심어 {sum(found.values())}건 → experiments/results/pseudonymize_report.md")
    if found:
        for k, v in found.most_common():
            print(f"  잔여: {k} ×{v}  …{examples[k]}…")
        sys.exit(1)


if __name__ == "__main__":
    main()
