"""온톨로지 파일럿 공용 모듈: 경로, 목록(질문 상황·Schwartz), ID 규칙, 데이터 읽기, 추출 프롬프트.

설계 기준: 인물 온톨로지 설계표 v0.3 (클래스·관계·공리·ID 규칙).
"""
import json, os, re, random, csv, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments/scripts"))

DATA = os.path.join(ROOT, "data/anon")
RUNS = os.path.join(ROOT, "experiments/runs/ontology")
PERSON = "P01"
PERSON_NAME = "노라"   # 가명본의 이름

SITUATIONS = ["조언 요청", "비판·반론 받음", "회상", "찬반 묻기", "칭찬 받음",
              "취향·습관 묻기", "근황·일상 대화", "기타"]
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


# ---------------- 추출 프롬프트 ----------------
EXTRACT_SYSTEM = """너는 인터뷰 발화를 인물 온톨로지의 칸으로 분해하는 코더다.
분류가 아니라 분해다: 한 답에서 입장이 여러 개 나올 수 있고, 말하지 않은 칸은 null로 둔다.
본인이 말하지 않은 이유·가치를 추측으로 채우지 않는다. 반드시 JSON 하나로만 답한다."""

EXTRACT_USER = """[질문]
{question}

[답]
{answer}

[질문 상황 목록] {situations}

[Schwartz 10칸]
{schwartz}

[지금까지의 세부 꼬리표 목록 — 같은 뜻이면 이 말을 그대로 쓰고, 없을 때만 새로 만든다]
{tags}

규칙
1. situation: 이 질문이 만든 상황. 목록에서 하나.
2. stances: 답에서 '무엇에 대해 찬성·반대하는지'가 드러나는 입장마다 하나. 없으면 빈 목록.
   - topic: 입장의 대상, 짧은 명사구 (예: "유행하는 옷")
   - polarity: 찬성 | 반대 | 조건부
   - attribute: 대상에 대해 본 특징. {{"tag": 꼬리표, "phrase": 답의 원문 구절}} — 말하지 않았으면 null
   - consequence: 그 특징이 가져오는 결과·이유 ("그래서 ~하다"). 같은 형식, 없으면 null
   - value: 결국 지키려는 것. {{"words": 본인의 말, "schwartz": 10칸 중 1~2개 목록, "effect": "지킴" 또는 "해침", "phrase": 원문 구절}}
            effect는 '대상이 그 가치를 지키는지 해치는지'. 본인이 이유를 말하지 않았으면 value 전체를 null.
   - evidence: 직접 진술 | 관찰 | 추론 (찬반을 본인이 말했으면 직접 진술, 말투로만 보이면 관찰, 짐작이면 추론)
   - topic_phrase: 대상과 찬반이 드러난 원문 구절
3. episodes: 본인이 겪은 구체적인 일(때·곳·사건이 있는 것)마다 {{"label": 짧은 이름, "phrase": 원문 구절}}. 없으면 빈 목록.
4. style: 이 답의 말투 한 줄 (길이, 단정성, 부르는 말 등).
5. 모든 phrase는 [답]에 실제로 있는 문자열을 그대로 복사한다. 꼬리표는 10자 안팎의 일반적인 말로.

JSON 형식:
{{"situation": "...", "stances": [{{"topic": "...", "polarity": "...", "attribute": {{"tag": "...", "phrase": "..."}} 또는 null, "consequence": {{"tag": "...", "phrase": "..."}} 또는 null, "value": {{"words": "...", "schwartz": ["..."], "effect": "...", "phrase": "..."}} 또는 null, "evidence": "...", "topic_phrase": "..."}}], "episodes": [], "style": "..."}}"""


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
            "evidence": r.choice(EVIDENCE), "topic_phrase": ph()})
    eps = [{"label": "모의 일화", "phrase": ph()}] if r.random() < 0.2 else []
    return {"situation": r.choice(SITUATIONS), "stances": stances, "episodes": eps, "style": "모의 말투"}
