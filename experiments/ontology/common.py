"""온톨로지 파일럿 공용 모듈: 경로, 목록(질문 상황·Schwartz), ID 규칙, 데이터 읽기, 추출 프롬프트.

설계 기준: 인물 온톨로지 설계표 v0.3 (클래스·관계·공리·ID 규칙).
"""
import json, os, re, random, csv, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments/scripts"))
CFG_MODELS = json.load(open(os.path.join(ROOT, "experiments/config/models.json"), encoding="utf-8"))

DATA = os.path.join(ROOT, "data/anon")
RUNS = os.path.join(ROOT, "experiments/runs/ontology")
PERSON = "P01"
PERSON_NAME = "노라"   # 가명본의 이름

SITUATIONS = ["조언 요청", "비판·반론 받음", "회상", "찬반 묻기", "칭찬 받음",
              "취향·습관 묻기", "근황·일상 대화", "기타"]
UNKNOWN_SITUATION = "알 수 없음"   # 원래 질문이 없는 문항(derived)
SCHWARTZ = {
    "자기주도": "스스로 생각하고 선택하고 만들고 탐구하는 독립성",
    "자극": "삶의 신남, 새로움, 도전",
    "쾌락": "나를 위한 즐거움과 감각적 만족",
    "성취": "사회 기준에 맞는 능력을 보여 얻는 개인적 성공",
    "권력": "사회적 지위와 위신, 사람과 자원에 대한 통제",
    "안전": "사회·관계·자신의 안전, 조화, 안정",
    "순응": "남을 불편하게 하거나 규범에 어긋날 행동·충동을 삼감",
    "전통": "문화·종교가 주는 관습과 생각을 존중하고 받아들임",
    "박애": "자주 만나는 가까운 사람들의 안녕을 지키고 높임",
    "보편주의": "모든 사람과 자연의 안녕을 이해·존중·보호",
}
SCHWARTZ_OK = set(SCHWARTZ) | {"해당 없음"}
POLARITY = ["찬성", "반대", "조건부"]
EVIDENCE = ["직접 진술", "관찰", "추론"]
EFFECT = ["지킴", "해침"]


# ---------------- 데이터와 ID ----------------
def load_split(name):
    return json.load(open(os.path.join(DATA, f"{name}.json"), encoding="utf-8"))["items"]


def video_index():
    """자료 ID(I번호): Train → Holdout → Retest 순, 같은 분할 안에서는 연도·video_id 순으로 고정 부여."""
    seen, order = set(), []
    for split in ["Train", "Holdout", "Retest"]:
        vids = {}
        for x in load_split(split):
            vids.setdefault(x["video_id"], x.get("year") or "")
        for v, y in sorted(vids.items(), key=lambda t: (t[1], t[0])):
            if v not in seen:
                seen.add(v); order.append(v)
    return {v: f"I{i+1:02d}" for i, v in enumerate(order)}


def statement_id(x, vidx):
    """발화 ID = 사람-자료-순번 (P01-I03-012). 순번은 원 데이터 id의 q번호."""
    m = re.search(r"_q(\d+)$", x["id"])
    seq = int(m.group(1)) if m else 0
    return f"{x.get('persona_code', PERSON)}-{vidx[x['video_id']]}-{seq:03d}"


CLEAN_DIR = os.path.join(RUNS, "_clean")


def load_clean():
    """clean.py 가 만든 교정본 {item_id: 교정 답}. 거절된 교정은 빠져 있다."""
    out = {}
    for split in ["Train", "Holdout", "Retest"]:
        p = os.path.join(CLEAN_DIR, f"{split}.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                if r.get("accepted"):
                    out[r["id"]] = r["clean"]
    return out


def run_dir(run):
    d = os.path.join(RUNS, run)
    os.makedirs(d, exist_ok=True)
    return d


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_csv(p, header, rows):
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------- 꼬리표 목록 ----------------
class TagBook:
    """세부 꼬리표 목록. 칸(A/C/V)별로 꼬리표와 사용 횟수를 들고 있다.
    --tags 로 코딩 시트의 '꼬리표' 탭을 CSV로 내려받아 넣으면 그 목록에서 시작한다."""

    SLOTS = {"A": "특징 A", "C": "결과 C", "V": "가치 V"}

    def __init__(self):
        self.tags = {"A": {}, "C": {}, "V": {}}

    @classmethod
    def from_csv(cls, path):
        tb = cls()
        if path and os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                slot = (r.get("칸") or "").strip()
                tag = (r.get("세부 꼬리표") or "").strip()
                state = (r.get("상태") or "").strip()
                if not tag or state in ("삭제", "예시"):
                    continue
                key = "A" if slot.startswith("특징") else "C" if slot.startswith("결과") else "V" if slot.startswith("가치") else None
                if key:
                    tb.tags[key].setdefault(tag, 0)
        return tb

    def add(self, key, tag):
        if tag:
            self.tags[key][tag] = self.tags[key].get(tag, 0) + 1

    def is_new(self, key, tag):
        return tag not in self.tags[key]

    def prompt_block(self, top=60):
        out = []
        for k, name in self.SLOTS.items():
            items = sorted(self.tags[k].items(), key=lambda t: -t[1])[:top]
            out.append(f"{name}: " + (", ".join(t for t, _ in items) if items else "(아직 없음)"))
        return "\n".join(out)

    def dump(self):
        return {k: dict(sorted(v.items(), key=lambda t: -t[1])) for k, v in self.tags.items()}


# ---------------- 추출 프롬프트 (v2) ----------------
# v1 → v2 (2026-11-22, 파일럿 1차 결과 반영)
#  - A·C·V의 뜻은 그대로 두고 경계를 분명히: 계기 사건 → 일화, 본인 성격 → 특질 단서(trait_cues), 말 안 했으면 비움
#  - 가치 칸: 정리된 짧은 말(words)과 원문 구절(phrase)을 분리
#  - 근거 등급을 찬반(evidence_polarity)과 가치(evidence_value)에 따로
#  - 말투 메모: 자막 인식 오류는 말투가 아님
#  - 만든 질문(derived)은 질문 상황을 '알 수 없음'으로 (코드에서 고정)
PROMPT_VERSION = "v2.1"  # v2.1: C에 Schwartz 가치를 그대로 쓰지 않는 규칙 (설계표 v0.4, A19)

EXTRACT_SYSTEM = """너는 인터뷰 발화를 인물 온톨로지의 칸으로 분해하는 코더다.
분류가 아니라 분해다: 한 답에서 입장이 여러 개 나올 수 있고, 말하지 않은 칸은 null로 둔다.
본인이 말하지 않은 이유·가치를 추측으로 채우지 않는다. 반드시 JSON 하나로만 답한다.
답은 자동 자막이라 틀린 단어·끊긴 말이 섞여 있다. 뜻을 읽되, 구절은 원문 그대로 복사한다."""

EXTRACT_USER = """[질문]
{question}

[답]
{answer}

[질문 상황 목록] {situations}

[Schwartz 10칸]
{schwartz}

[지금까지의 세부 꼬리표 목록 — 같은 뜻이면 이 말을 그대로 쓰고, 없을 때만 새로 만든다]
{tags}

칸의 뜻 (수단-목적 사슬: 대상 → A → C → V)
- A 특징: "그것(대상)이 어떤 것이라고 보나?" 대상의 성질이나 그 선택에 대한 규정.
    물건이면 "한철이다", "손이 많이 간다" / 선택·행동이면 "남들이 안 가는 길", "늦어도 할 수 있는 일"
    A에 넣지 않는 것: 계기가 된 사건(→ episodes), 본인의 성격(→ trait_cues), 본인의 감정·상태
- C 결과: "그래서 나에게 무엇이 생기나?" 그 대상·특징 때문에 본인에게 생기는 일. 예: "오래 못 입는다", "후회가 줄어든다"
    C 꼬리표에 Schwartz 가치의 이름이나 그 뜻("안전해진다", "자유로워진다")을 그대로 쓰지 않는다. 그런 말은 value에 넣는다.
- V 가치: "그게 왜 중요한가?" 사슬의 끝. words는 정리된 짧은 말(10자 안팎, 자막 오류 없이), phrase는 원문 구절.

규칙
1. situation: 이 질문이 만든 상황. 목록에서 하나.
2. stances: 답에서 '무엇에 대해 찬성·반대하는지'가 드러나는 입장마다 하나. 없으면 빈 목록.
   - topic: 입장의 대상, 짧은 명사구
   - polarity: 찬성 | 반대 | 조건부
   - attribute: {{"tag": 꼬리표, "phrase": 원문 구절}} 또는 null
   - consequence: {{"tag": 꼬리표, "phrase": 원문 구절}} 또는 null
   - value: {{"words": 정리된 말, "schwartz": 10칸 중 1~2개, "effect": "지킴" 또는 "해침", "phrase": 원문 구절}} 또는 null
            effect는 '대상이 그 가치를 지키는지 해치는지'. 이유가 전혀 없으면 value 전체를 null.
   - evidence_polarity: 찬반의 근거 — 직접 진술(본인이 찬반을 말함) | 관찰(말투·태도로만 보임) | 추론(짐작)
   - evidence_value: 가치의 근거 — 직접 진술(본인이 '왜 중요한지'를 말함) | 추론(답에서 짐작) | value가 null이면 null
     Schwartz 칸을 고른 것 자체는 해석이므로, 본인이 이유를 분명히 말하지 않았다면 '추론'이다.
   - topic_phrase: 대상과 찬반이 드러난 원문 구절
3. episodes: 본인이 겪은 구체적인 일(때·곳·사건이 있는 것)마다 {{"label": 짧은 이름, "phrase": 원문 구절}}. 계기가 된 사건도 여기.
4. trait_cues: 답에서 드러난 본인의 성격 단서(예: "책임감", "실행력") 목록. 없으면 빈 목록.
5. style: 말투 한 줄 — 어휘, 문장 길이, 단정성, 부르는 말만. 자막 인식 오류(끊김, 틀린 단어, 반복)는 말투로 적지 않는다.
6. 모든 phrase는 [답]에 실제로 있는 문자열을 그대로 복사한다. 꼬리표는 10자 안팎의 일반적인 말로.

JSON 형식:
{{"situation": "...", "stances": [{{"topic": "...", "polarity": "...", "attribute": {{"tag": "...", "phrase": "..."}} 또는 null, "consequence": {{"tag": "...", "phrase": "..."}} 또는 null, "value": {{"words": "...", "schwartz": ["..."], "effect": "...", "phrase": "..."}} 또는 null, "evidence_polarity": "...", "evidence_value": "..." 또는 null, "topic_phrase": "..."}}], "episodes": [], "trait_cues": [], "style": "..."}}"""


def extract_prompt(question, answer, tagbook):
    sch = "\n".join(f"- {k}: {v}" for k, v in SCHWARTZ.items()) + "\n- 해당 없음: 10칸 어디에도 안 맞음"
    return EXTRACT_USER.format(question=question, answer=answer, situations=" | ".join(SITUATIONS),
                               schwartz=sch, tags=tagbook.prompt_block())


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def mock_extract(question, answer, seed=0):
    """MOCK_LLM=1 일 때 쓰는 가짜 추출 결과 (형식만 맞춤, 파이프라인 점검용)."""
    r = random.Random(f"{seed}{answer[:30]}")
    words = [w for w in re.findall(r"[가-힣]{2,}", answer)] or ["무엇"]
    def ph():
        i = r.randrange(len(words)); return " ".join(words[i:i + 2])
    stances = []
    for _ in range(r.choice([0, 1, 1, 2])):
        stances.append({
            "topic": r.choice(words), "polarity": r.choice(POLARITY),
            "attribute": r.choice([None, {"tag": r.choice(["오래가지 않는다", "남과 같아진다", "정성이 든다", "비싸다"]), "phrase": ph()}]),
            "consequence": r.choice([None, {"tag": r.choice(["쓸모가 금방 끝난다", "마음이 편하다", "내가 드러나지 않는다"]), "phrase": ph()}]),
            "value": r.choice([None, None, {"words": r.choice(["오래가는 것", "나답게", "나누는 삶"]), "schwartz": [r.choice(list(SCHWARTZ))], "effect": r.choice(EFFECT), "phrase": ph()}]),
            "evidence_polarity": r.choice(EVIDENCE), "evidence_value": r.choice(["직접 진술", "추론"]), "topic_phrase": ph()})
    eps = [{"label": "모의 일화", "phrase": ph()}] if r.random() < 0.2 else []
    return {"situation": r.choice(SITUATIONS), "stances": stances, "episodes": eps, "trait_cues": ["모의 성격"], "style": "모의 말투"}
