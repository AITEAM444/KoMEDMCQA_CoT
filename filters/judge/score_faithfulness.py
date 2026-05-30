"""
G-Eval 방식 CoT 품질 채점 — 파이프라인 F8 (TAIL) 용.

차원 3개 (각 1~5점, 독립 평가):
    - Step Coherence  : 단계 간 논리적 연결 (reference-guided)
    - Korean Fluency  : 번역체 없는 자연스러운 한국어 의학 표현
    - Step Coverage   : Step 1~4 각 역할 실질 수행 여부

집계: min(3차원). 루브릭·프롬프트는 data/evaluated/score_faithfulness.py v3과 동일.
"""

import json
import os
import re
import sys
import time

from openai import OpenAI


SOLAR_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = os.environ.get("SOLAR_MODEL", "solar-pro3")

DIMENSIONS = {
    "step_coherence": {
        "dim_name": "Step Coherence (단계 간 논리적 일관성)",
        "definition": "Step 2 원리 -> Step 3 선택지 판단 -> Step 4 결론의 인과 사슬이 일관적인가",
        "calibration_note": (
            "정답은 직접 가점하지 않는다. 다만 Step 3에서 부적합하다고 판단한 선택지를 "
            "Step 4에서 정답으로 고르는 직접 모순을 찾기 위해서만 사용한다."
        ),
        "reference_guided": True,
        "evidence_to_collect": (
            "E1. Step 3에서 부적합으로 판정한 선택지가 Step 4 정답으로 선택되었는가? "
            "정답 선택지는 {gold}이다. (Y/N)\n"
            "E2. Step 2의 의학 원리가 Step 3 각 선택지 판단에 명시적으로 인용/적용되었는가? "
            "(전체/부분/없음)\n"
            "E3. Step 3 또는 Step 4에 Step 2 원리로 설명 안 되는 새 주장이 등장하는가? (개수)\n"
            "E4. 정답 선택지에 대한 핵심 근거가 Step 2 또는 Step 3에 명시되었는가? (Y/N)"
        ),
        "anchors": (
            "5점: E1=N, E2=전체, E3=0, E4=Y. 사슬 완결.\n"
            "4점: E1=N이고 E4=Y이지만, E2=부분 또는 E3=1건의 경미한 비약.\n"
            "3점: E1=N이지만 E3>=2건이거나, E4가 약함(근거가 암시적).\n"
            "2점: E1=N이지만 E2=없음 또는 E3>=3건. 결론이 사슬과 무관하게 도출됨.\n"
            "1점: E1=Y. Step 간 직접적 모순."
        ),
    },
    "korean_fluency": {
        "dim_name": "Korean Fluency (한국어 의료 표현의 자연스러움)",
        "definition": "번역체, 불필요한 영어 혼용, 한국 임상 표현과의 불일치가 없는가",
        "calibration_note": "한국 의료진이 읽었을 때 자연스러운 임상 한국어인지 평가한다.",
        "reference_guided": False,
        "evidence_to_collect": (
            "E1. 불필요한 영어 병기 정도 (거의 없음 / 약간 / 빈번). "
            "단, CT, MRI, COPD 등 한국 임상 표준 약어는 제외. "
            "개별 영어 단어를 나열하지 말 것. 정도만 한 단어로 판정하라.\n"
            "E2. 번역체 문장 정도 (없음 / 소수 / 다수). "
            "예: 직역투, 영어 어순, '~에 대하여', '~을 가지고 있다'. "
            "문장을 인용하지 말 것. 정도만 판정하라.\n"
            "E3. 한국어 문장 안에 영어 구문이 통째로 삽입된 사례 (없음 / 있음 + 개수).\n"
            "E4. 의학 용어가 한국 표준 용어와 일치하는가? (일치 / 불일치)."
        ),
        "anchors": (
            "5점: E1=거의 없음, E2=없음, E3=없음, E4=일치. 한국 의료진이 쓴 글과 구별 불가.\n"
            "4점: E1=약간, E2=없음~소수, E3=없음, E4=일치. 가벼운 영어 병기만 존재.\n"
            "3점: E1=약간~빈번 또는 E2=소수, E3=없음~1건. 번역체가 감지되나 의미 전달은 명확.\n"
            "2점: E1=빈번 또는 E2=다수 또는 E3>=2. 번역체가 두드러져 한국 임상 글로 부자연스러움.\n"
            "1점: E4=불일치 또는 E3>=3."
        ),
    },
    "step_coverage": {
        "dim_name": "Step Coverage (각 Step의 실질 수행)",
        "definition": "Step 1~4 각각이 라벨에 해당하는 역할을 실질적으로 수행했는가",
        "calibration_note": "단계 제목만 있는 형식적 작성은 수행으로 보지 않는다.",
        "reference_guided": False,
        "evidence_to_collect": (
            "E1. Step 1이 문제의 핵심 임상 정보와 조건을 실제로 요약했는가, 단순 재진술인가? "
            "(실질/재진술)\n"
            "E2. Step 2가 적용 의학 원리를 구체적으로 서술했는가, 일반론인가? (구체/일반)\n"
            "E3. Step 3가 5개 선택지 모두를 의학적 근거로 검토했는가? (검토한 선택지 개수, 0~5)\n"
            "E4. Step 4가 결론과 그 이유를 명시했는가, 정답만 반복인가? (이유 명시/반복)"
        ),
        "anchors": (
            "5점: E1=실질, E2=구체, E3=5, E4=이유 명시. 4단계 모두 충실.\n"
            "4점: E3=5이지만 E1, E2, E4 중 하나가 약함.\n"
            "3점: E3=4 또는 두 단계가 형식만 채움.\n"
            "2점: E3=2~3 또는 세 단계가 형식만 채움.\n"
            "1점: E3<=1 또는 Step 자체가 빠짐."
        ),
    },
}

DIM_ORDER = ["step_coherence", "korean_fluency", "step_coverage"]

REFERENCE_INSTRUCTION = {
    True:  "정답은 증거 수집 항목에 제공되어 있으나, 정답 일치 자체를 가점하지 마세요. "
           "'근거가 부실한데 우연히 정답에 도달한' 경우를 적발하는 용도로만 사용하세요.",
    False: "정답 여부는 평가하지 말고, 응답 텍스트 안에서 직접 확인되는 추론 품질만 평가하세요.",
}

GEVAL_PROMPT = """[Task Introduction]
당신은 한국 의료 자격시험 CoT(Chain-of-Thought)의 품질을 평가하는 의료 전문가입니다.
아래 단일 차원만 독립적으로 평가하세요.

[Evaluation Criteria]
차원: {dim_name}
정의: {definition}
보정 지침: {calibration_note}
정답 참조 지침: {reference_instruction}

[Evidence To Collect]
점수보다 먼저 아래 E1~E4 증거를 관찰하세요.
{evidence_to_collect}

[Anchors]
{anchors}

[General Principles]
- 경계가 애매하면 하향 채점.
- "전반적으로 좋다", "전반적으로 자연스럽다" 같은 인상 평가 금지.
- 반드시 응답 텍스트에서 직접 확인되는 증거에 근거.
- 5점은 "완벽"이 아니라 "감점 사유 0개"를 의미.

[Input]
문제:
{question}

선지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

[CoT]
{reasoning}

[Output]
E1: <관찰>
E2: <관찰>
E3: <관찰>
E4: <관찰>
앵커 적용: <어느 앵커 부합 + 근거 1문장>
마지막 줄에는 반드시 다음 JSON 한 줄만 출력:
{{"score": <int>, "reason": "<한 문장>"}}
"""


def parse_geval_response(text):
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    candidates = re.findall(r'\{[^{}]*"score"\s*:\s*[0-9]+[^{}]*\}', text)
    if candidates:
        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
                if isinstance(obj.get("score"), (int, float)) and 1 <= obj["score"] <= 5:
                    return obj
            except json.JSONDecodeError:
                continue

    m = re.findall(r'"score"\s*:\s*([1-5])', text)
    if m:
        return {"score": int(m[-1]), "reason": "recovered_from_broken_json"}

    m = re.findall(r'(\d)\s*점\s*(?:기준|에 해당|입니다|[:.])', text)
    if m:
        sc = int(m[-1])
        if 1 <= sc <= 5:
            return {"score": sc, "reason": "recovered_from_korean_text"}

    return None


def make_client():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        print("ERROR: UPSTAGE_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=SOLAR_BASE_URL)


def call_judge(client, prompt, temperature):
    response = client.chat.completions.create(
        model=SOLAR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def score_one_dim(client, sample_row, result, dim_key, reps, temperature, sleep_on_error):
    spec = DIMENSIONS[dim_key]

    evidence = spec["evidence_to_collect"]
    if spec.get("reference_guided"):
        gold = result.get("correct", "")
        evidence = evidence.format(gold=gold)

    ref_instruction = REFERENCE_INSTRUCTION[spec.get("reference_guided", False)]
    reasoning = result.get("final_content") or result.get("reasoning") or ""

    prompt = GEVAL_PROMPT.format(
        dim_name=spec["dim_name"],
        definition=spec["definition"],
        calibration_note=spec["calibration_note"],
        evidence_to_collect=evidence,
        anchors=spec["anchors"],
        reference_instruction=ref_instruction,
        question=sample_row["question"],
        A=sample_row["A"],
        B=sample_row["B"],
        C=sample_row["C"],
        D=sample_row["D"],
        E=sample_row["E"],
        reasoning=reasoning[:8000],
    )

    runs = []
    raws = []
    reasons = []
    errors = []
    for _ in range(reps):
        text = None
        parsed = None
        last_err = None
        for _ in range(3):
            try:
                text = call_judge(client, prompt, temperature)
                parsed = parse_geval_response(text)
                if parsed is not None:
                    break
                last_err = "empty_response" if not text or not text.strip() else "parse_failed"
                if sleep_on_error:
                    time.sleep(sleep_on_error)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if sleep_on_error:
                    time.sleep(sleep_on_error)
        raws.append(text)
        if parsed is not None:
            runs.append(int(parsed["score"]))
            reasons.append(parsed.get("reason", ""))
            errors.append(None)
        else:
            runs.append(None)
            errors.append(last_err)
            reasons.append(None)

    valid_runs = [r for r in runs if r is not None]
    avg = sum(valid_runs) / len(valid_runs) if valid_runs else None
    return {
        "runs": valid_runs,
        "mean": avg,
        "reasons": reasons,
        "raws": raws,
        "errors": errors,
    }
