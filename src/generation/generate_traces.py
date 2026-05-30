"""
KorMedMCQA — 형식 강제 CoT 프롬프트로 Teacher 후보를 평가/생성.

본 스크립트는 두 가지 용도로 사용된다:

  1) Phase 0 — Teacher 후보 파일럿 비교
     GPT-4 / GPT-4o / DeepSeek-R1 등 후보 모델에 동일 프롬프트(형식 강제)를
     적용해 4-Step 출력 품질을 비교. 보고서 §3.2 에 따라 Teacher 는 이미
     **DeepSeek-R1 으로 확정**되었으므로 본 단계는 *역사적 비교 자료* 다.

  2) Phase 1 — Train split 전체 Teacher CoT 생성
     확정 Teacher(DeepSeek-R1)로 Train split 전체의 Raw CoT 를 생성.
     반드시 `--split train` 으로 실행해야 한다.

──────────────────────────────────────────────────────────────────────────────
중요 (보고서 §2): Train/Test 분할은 모든 Phase 의 절대 조건.
  - Phase 1 Train CoT 생성에 test split 사용 금지.
  - Phase 0 파일럿도 가능하면 train split 사용 권장 (보고서 §2 경고).
  - test split 은 마지막 KorMedMCQA Test 평가지표(Phase 6)에 사용한다.
──────────────────────────────────────────────────────────────────────────────

OpenAI 외 모델(예: DeepSeek-R1) 호출은 별도 클라이언트/엔드포인트 설정이
필요하다 (DeepSeek API 또는 OpenRouter 등). 본 스크립트는 OpenAI 호환
client 만 내장하므로 비-OpenAI Teacher 사용 시 client 교체 후 실행한다.

사용:
    # Phase 0 GPT 후보 비교 (역사적)
    python eval_cot.py --model gpt-4   --total 100 --split test
    python eval_cot.py --model gpt-4o  --total 100 --split test

    # Phase 1 Train CoT 생성 (DeepSeek-R1 client 설정 후)
    python eval_cot.py --model deepseek-r1 --total -1 --split train \\
        --output data/raw/cot_train_generated.jsonl
        
    $env:DEEPSEEK_API_KEY = "sk-..."   # 먼저 키 설정

    # Phase 1 Train 전체 생성, 16개 동시 호출
    py scripts\eval_cot.py --model deepseek-r1 --total -1 --split train `
    --workers 16 --output data\raw\cot_train_generated.jsonl

    # 중단 후 재개: 같은 명령을 다시 실행하면 출력 파일의 성공분은 건너뛴다
    # (실패/빈 응답만 재시도). 처음부터 다시 하려면 --overwrite.

    # 샤딩(여러 터미널 분산): N개 터미널에서 i=0..N-1 로 각각 실행
    #   py scripts\eval_cot.py ... --shard 0/4 --output data\raw\cot_train.jsonl
    #   py scripts\eval_cot.py ... --shard 1/4 --output data\raw\cot_train.jsonl  (다른 터미널)
    # → cot_train.shard0-of-4.jsonl ... 생성. 병합:
    #   Get-Content data\raw\cot_train.shard*-of-4.jsonl | Set-Content data\raw\cot_train.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# langdetect 는 무지시(free) 모드에서 think 언어 분포 기록에만 쓰인다.
# 미설치 시 forced 모드 동작은 막지 않고, 언어 판별만 비활성화한다.
try:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # langdetect 자체 무작위성 제거 (재현성)
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False


SUBSETS = ["dentist", "doctor", "nurse", "pharm"]

# DeepSeek API 모델 ID 매핑 (--model 에 쓰는 별칭 → 실제 API 모델 id).
DEEPSEEK_MODEL_IDS = {"deepseek-r1": "deepseek-reasoner"}

PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    # 비-OpenAI Teacher: 별도 client 설정 시 요금은 호출처에서 확인.
    "deepseek-r1": {"input": 0.0, "output": 0.0},
}


PROMPT_TEMPLATE = """다음은 한국 의료 자격시험(KorMedMCQA) 객관식 문제입니다.
당신은 의사·치과의사·약사·간호사 수준의 의료 전문가입니다.
아래 [출력 형식]을 반드시 그대로 따라 최종 답변을 한국어로 작성하세요.

[문제]
{question}

[선택지]
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

[출력 형식 — 반드시 아래 4개 Step + 마지막 줄 형태로 출력]
[Step 1] 환자/문제 상황 정리: (문제에서 주어진 핵심 임상 정보·조건을 한국어로 요약)
[Step 2] 관련 의학적 원리·지식: (이 문제에 적용되는 해부·생리·병태생리·약리·진단·치료 원리를 서술)
[Step 3] 선택지별 검토: A부터 E까지 각 선택지를 한 줄씩이라도 검토하여 적합/부적합 사유를 명시
[Step 4] 결론 도출: 위 근거를 종합하여 가장 적절한 답을 고른 이유를 한 문단으로 정리
정답: X

[제약 — 위반 금지]
1. 위 4개의 Step 라벨([Step 1], [Step 2], [Step 3], [Step 4])과 "정답: X" 라인을 모두 빠짐없이 출력합니다.
2. "단계별 추론을 제공할 수 없다", "답할 수 없다", "간단한 근거" 등 형식 거부·축약 응답은 금지입니다. 불확실해도 가장 가능성이 높은 선택지를 골라 위 형식대로 작성하세요.
3. 마지막 줄은 정확히 "정답: X" (X는 A, B, C, D, E 중 하나)여야 합니다. 괄호·따옴표·마침표·추가 텍스트를 붙이지 마세요.
4. [Step 1]~[Step 4]와 "정답: X" 줄, 즉 최종 답변은 모두 한국어로 작성합니다.
"""


# 무지시(free) 프롬프트: 언어·4-Step 형식을 강제하지 않는다. think 가 어떤
# 언어로 나오는지는 막지 않고, 답 추출을 위해 마지막 "정답: X" 줄만 요청한다.
# (Qi et al. §2.2 "무지시 시 영어 회귀" 현상이 한국어 의료 데이터에서도 나타나는지
#  reasoning_content 언어 분포로 직접 확인하기 위한 비교군.)
FREE_PROMPT_TEMPLATE = """다음은 한국 의료 자격시험(KorMedMCQA) 객관식 문제입니다.
문제를 풀고 가장 적절한 정답을 고르세요.

[문제]
{question}

[선택지]
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

마지막 줄에 "정답: X" (X는 A~E) 형식으로 적으세요.
"""

PROMPT_TEMPLATES = {"forced": PROMPT_TEMPLATE, "free": FREE_PROMPT_TEMPLATE}


# fewshot 모드: 영어 instruction + 분야별 한국어 fewshot(데이터셋 fewshot split, 5개).
# 내부추론(reasoning_content)의 언어를 *강제하지 않고* 관찰하기 위한 설정 — instruction 은
# 추론 방식·언어를 일절 지시하지 않으며, fewshot 만 한국어 예시로 노출한다.
FEWSHOT_INSTRUCTION = """You are a medical expert taking the Korean medical licensing examination.
Read the question and choose the single best answer based on careful clinical reasoning.
Write the final answer on the last line exactly as: 정답: X (X is one of A, B, C, D, E).
Do not write anything after the answer line.
===== Example =====
{fewshot}
===== Now answer the following =====
{question_block}"""


def _format_q_block(sample):
    return (
        f"[문제]\n{sample['question']}\n\n[선택지]\n"
        f"A. {sample['A']}\nB. {sample['B']}\nC. {sample['C']}\n"
        f"D. {sample['D']}\nE. {sample['E']}"
    )


def _format_fewshot_example(row):
    cot = (row.get("cot") or "").strip()
    return f"{_format_q_block(row)}\n\n{cot}\n정답: {label_to_letter(row['answer'])}"


_FEWSHOT_CACHE = {}


def load_fewshot_blocks(subsets):
    """분야별 fewshot(5개) 을 하나의 텍스트 블록으로 캐싱해 반환. {subset: block}."""
    for subset in subsets:
        if subset in _FEWSHOT_CACHE:
            continue
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="fewshot")
        _FEWSHOT_CACHE[subset] = "\n\n---\n\n".join(
            _format_fewshot_example(row) for row in ds
        )
    return _FEWSHOT_CACHE


def build_prompt(sample, prompt_mode="forced", fewshot_blocks=None):
    if prompt_mode == "fewshot":
        subset = sample.get("subject") or sample.get("subset")
        return FEWSHOT_INSTRUCTION.format(
            fewshot=fewshot_blocks[subset],
            question_block=_format_q_block(sample),
        )
    return PROMPT_TEMPLATES[prompt_mode].format(
        question=sample["question"],
        A=sample["A"],
        B=sample["B"],
        C=sample["C"],
        D=sample["D"],
        E=sample["E"],
    )


def detect_think_language(text, mixed_threshold=0.2):
    """reasoning_content(<think>)의 언어를 korean/english/mixed/other 로 분류.

    langdetect.detect_langs 의 언어별 확률을 사용한다. 한국어·영어가 모두
    mixed_threshold 이상이면 'mixed', 아니면 우세 언어. langdetect 미설치 시
    label=None (판별 생략), 텍스트가 너무 짧아 판별 실패 시 'undetermined'.
    """
    if not _LANGDETECT_AVAILABLE or not text or not text.strip():
        return {"label": None, "probs": None}
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return {"label": "undetermined", "probs": None}
    probs = {lang.lang: round(lang.prob, 4) for lang in langs}
    ko = probs.get("ko", 0.0)
    en = probs.get("en", 0.0)
    if ko >= mixed_threshold and en >= mixed_threshold:
        label = "mixed"
    elif ko > 0 and ko >= en:
        label = "korean"
    elif en > 0:
        label = "english"
    else:
        label = "other"
    return {"label": label, "probs": probs}


_ANSWER_LINE_RE = re.compile(r"정답\s*[::]\s*\(?\s*([A-E])\s*\)?")


def parse_answer(response_text):
    if not response_text:
        return None
    matches = _ANSWER_LINE_RE.findall(response_text)
    if matches:
        return matches[-1]
    # 폴백: 마지막 80자에서 가장 뒤에 등장한 A-E 한 글자 사용.
    # (이전 구현은 ["A","B","C","D","E"] 순서로 탐색해 A 편향이 있었다.)
    last_chunk = response_text.strip()[-80:]
    for char in reversed(last_chunk):
        if char in "ABCDE":
            return char
    return None


def check_format(response_text):
    """형식 강제가 잘 지켜졌는지 점검 (faithfulness 채점의 구조적 부분)."""
    if not response_text:
        return {
            "has_step1": False,
            "has_step2": False,
            "has_step3": False,
            "has_step4": False,
            "has_answer_line": False,
            "all_steps_present": False,
        }
    has_s1 = "[Step 1]" in response_text
    has_s2 = "[Step 2]" in response_text
    has_s3 = "[Step 3]" in response_text
    has_s4 = "[Step 4]" in response_text
    has_ans = bool(_ANSWER_LINE_RE.search(response_text))
    return {
        "has_step1": has_s1,
        "has_step2": has_s2,
        "has_step3": has_s3,
        "has_step4": has_s4,
        "has_answer_line": has_ans,
        "all_steps_present": all([has_s1, has_s2, has_s3, has_s4, has_ans]),
    }


def label_to_letter(label):
    return chr(ord("A") + int(label) - 1)


def evaluate_one(client, model, sample, prompt_mode="forced", seed=42, fewshot_blocks=None):
    prompt = build_prompt(sample, prompt_mode, fewshot_blocks)
    correct_letter = label_to_letter(sample["answer"])
    api_model = DEEPSEEK_MODEL_IDS.get(model, model)

    try:
        # 디코딩 고정 (재현성): temperature=0 (greedy) + seed 고정.
        # 주의: deepseek-reasoner(R1)는 API 상 temperature/seed 를 무시할 수
        #       있어 완전한 결정성은 보장되지 않는다 (best-effort).
        response = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=seed,
            max_tokens=8000,
        )
        message = response.choices[0].message
        response_text = message.content
        # DeepSeek-R1 은 <think> 사고과정을 reasoning_content 로 분리 반환한다.
        reasoning_content = getattr(message, "reasoning_content", None)
        predicted = parse_answer(response_text)
        fmt = check_format(response_text)
        think_lang = detect_think_language(reasoning_content)

        return {
            "predicted": predicted,
            "correct": correct_letter,
            "is_correct": predicted == correct_letter,
            "reasoning": response_text,
            "reasoning_content": reasoning_content,
            "format_check": fmt,
            "prompt_mode": prompt_mode,
            "seed": seed,
            "think_lang": think_lang["label"],
            "think_lang_probs": think_lang["probs"],
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "predicted": None,
            "correct": correct_letter,
            "is_correct": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "prompt_mode": prompt_mode,
            "seed": seed,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def samples_per_subset(total_samples, subsets):
    base = total_samples // len(subsets)
    remainder = total_samples % len(subsets)
    return {
        subset: base + (1 if i < remainder else 0)
        for i, subset in enumerate(subsets)
    }


def load_samples(total_samples, split="test", subsets=None):
    """KorMedMCQA 샘플 로드.

    Args:
        total_samples: 총 샘플 수. -1 이면 split 전체 사용.
        split: "train" / "test" / "fewshot" / "dev".
            * Phase 1 Teacher CoT 생성: "train" (보고서 §2 절대 조건)
            * Phase 0 파일럿: 역사적으로 "test" 사용. 가능하면 "train" 권장.
        subsets: 사용할 분야 목록. None 이면 4분야 전체. (--subset 으로 한 분야만 실행 가능)
    """
    if split == "test":
        print("⚠ test split 사용 — Phase 1 Teacher CoT 생성에는 금지 (보고서 §2).")

    subsets = subsets or SUBSETS
    use_all = total_samples is not None and total_samples < 0
    requested = (
        None if use_all else samples_per_subset(total_samples, subsets)
    )
    all_samples = []

    print(f"데이터 로딩 중... (split={split})")
    for subset in subsets:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split=split)
        count = len(ds) if use_all else min(requested[subset], len(ds))
        selected = ds.select(range(count))
        for sample_idx, sample in enumerate(selected):
            all_samples.append(
                {"subset": subset, "sample_idx": sample_idx, "sample": sample}
            )
        print(f"- {subset}: {count}건")

    return all_samples


def parse_shard(shard_str):
    """'i/N' → (i, N). 형식·범위 검증 (0 <= i < N, N >= 1). None 이면 비활성."""
    if shard_str is None:
        return None
    try:
        i_str, n_str = shard_str.split("/")
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise SystemExit(f"--shard 형식 오류: 'i/N' 이어야 함 (받은 값: {shard_str!r})")
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(
            f"--shard 범위 오류: 0 <= i < N 이고 N >= 1 (받은 값: {shard_str!r})"
        )
    return i, n


def shard_output_path(output_file, shard):
    """샤드 실행 시 출력 파일명에 '.shardI-of-N' 삽입 (확장자 보존)."""
    if shard is None:
        return output_file
    i, n = shard
    root, ext = os.path.splitext(output_file)
    return f"{root}.shard{i}-of-{n}{ext}"


def load_done(path):
    """기존 출력에서 성공 완료된 결과를 회수 → {global_idx: record}.

    체크포인트 정책: error 가 있거나 응답(reasoning)이 빈 줄은 재시도 대상이므로
    완료로 치지 않는다. 중단으로 잘린 마지막 줄(JSON 파싱 실패)도 무시한다.
    """
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") or not rec.get("reasoning"):
                continue
            gid = rec.get("global_idx")
            if gid is not None:
                done[gid] = rec
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="예: gpt-4, gpt-4o, deepseek-r1 (확정 Teacher)")
    parser.add_argument("--total", type=int, default=100,
                        help="총 샘플 수. -1 이면 split 전체 (Phase 1 Train 생성용).")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "fewshot", "dev"],
                        help="KorMedMCQA split. Phase 1 Train CoT 생성은 'train' 필수.")
    parser.add_argument("--prompt-mode", dest="prompt_mode",
                        default="forced", choices=["forced", "free", "fewshot"],
                        help="forced=한국어 4-Step 강제, free=무지시(언어/형식 자유), "
                             "fewshot=영어 instruction+분야별 한국어 fewshot(내부추론 관찰용).")
    parser.add_argument("--subset", default=None, choices=SUBSETS,
                        help="한 분야만 생성 (4분야 따로 돌릴 때). 생략 시 4분야 전체.")
    parser.add_argument("--seed", type=int, default=42,
                        help="디코딩 seed 고정 (재현성). temperature 는 0(greedy).")
    parser.add_argument("--workers", type=int, default=1,
                        help="동시 API 호출 수 (ThreadPoolExecutor). API 응답 대기가 "
                             "병목이므로 N 배 가속. rate limit 안에서 조절 (예: 16). "
                             "기본 1 = 순차.")
    parser.add_argument("--shard", default=None,
                        help="'i/N' 형식. 전체 샘플을 N 등분 중 i 번째(0-base)만 처리. "
                             "여러 터미널에서 i=0..N-1 로 분산 실행 후 JSONL 병합. "
                             "global_idx 는 전체 기준으로 부여되어 샤드 간 충돌 없음.")
    parser.add_argument("--overwrite", action="store_true",
                        help="기본은 기존 출력 파일이 있으면 이어서 진행(체크포인트). "
                             "이 플래그를 주면 처음부터 새로 생성.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model = args.model
    total = args.total
    prompt_mode = args.prompt_mode
    seed = args.seed
    workers = max(1, args.workers)
    shard = parse_shard(args.shard)
    base_output = args.output or f"results_{model}_cot_{prompt_mode}_{total}.jsonl"
    output_file = shard_output_path(base_output, shard)

    # 비-OpenAI Teacher(DeepSeek-R1)는 별도 엔드포인트로 호출한다 (CLAUDE.md 참조).
    if model.startswith("deepseek"):
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    else:
        client = OpenAI()

    mode_label = "형식 강제 CoT" if prompt_mode == "forced" else "무지시(free) CoT"
    print(f"=== KorMedMCQA {args.split} - {model} 평가 ({mode_label}) ===")
    print(f"디코딩: temperature=0 (greedy), seed={seed}")
    print(f"동시 호출(workers): {workers}")
    if shard is not None:
        print(f"샤드: {shard[0]}/{shard[1]} (global_idx % {shard[1]} == {shard[0]})")
    if not _LANGDETECT_AVAILABLE:
        print("⚠ langdetect 미설치 — think 언어 분포 기록이 비활성화됩니다 "
              "(pip install langdetect).")
    print()

    run_subsets = [args.subset] if args.subset else SUBSETS
    samples = load_samples(total, split=args.split, subsets=run_subsets)

    # fewshot 모드면 분야별 fewshot 블록을 미리 로드 (분야별 5개, 데이터셋 fewshot split).
    fewshot_blocks = load_fewshot_blocks(run_subsets) if prompt_mode == "fewshot" else None
    if prompt_mode == "fewshot":
        print(f"fewshot 로드: { {s: '5개' for s in run_subsets} }")

    # global_idx 는 전체 샘플 기준으로 먼저 부여 → 샤드와 무관하게 안정적 id.
    indexed = list(enumerate(samples))
    if shard is not None:
        i, n = shard
        indexed = [(g, it) for (g, it) in indexed if g % n == i]

    # 체크포인트: 기존 출력의 성공 완료분을 회수해 건너뛴다 (--overwrite 시 무시).
    done = {} if args.overwrite else load_done(output_file)
    pending = [(g, it) for (g, it) in indexed if g not in done]
    if done:
        print(f"체크포인트: 기존 {len(done)}건 완료 → 건너뜀, {len(pending)}건 남음")

    print(f"\n이번 실행 평가 대상: {len(pending)}건 "
          f"(샤드 내 전체 {len(indexed)}건)\n")

    correct_count = 0
    error_count = 0
    format_ok_count = 0
    total_input = 0
    total_output = 0
    per_subset = {s: {"total": 0, "correct": 0, "errors": 0} for s in SUBSETS}
    lang_counter = Counter()  # think(reasoning_content) 언어 분포

    def run_one(global_idx, item):
        """워커 스레드에서 실행 — API 호출만 담당. 집계/쓰기는 메인에서."""
        sample = item["sample"]
        result = evaluate_one(client, model, sample, prompt_mode, seed, fewshot_blocks)
        result["global_idx"] = global_idx
        result["subset"] = item["subset"]
        result["sample_idx"] = item["sample_idx"]
        result["question"] = sample["question"][:100] + "..."
        result["model"] = model
        return result

    def aggregate_and_write(result, f):
        """메인 스레드에서만 호출 — 통계 집계 + JSONL 쓰기 (락 불필요)."""
        nonlocal correct_count, error_count, format_ok_count
        nonlocal total_input, total_output
        subset = result["subset"]
        per_subset[subset]["total"] += 1
        if result.get("error"):
            error_count += 1
            per_subset[subset]["errors"] += 1
        else:
            if result["is_correct"]:
                correct_count += 1
                per_subset[subset]["correct"] += 1
            if result.get("format_check", {}).get("all_steps_present"):
                format_ok_count += 1
            if result.get("think_lang"):
                lang_counter[result["think_lang"]] += 1

        total_input += result.get("input_tokens", 0)
        total_output += result.get("output_tokens", 0)

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    # 파일은 새로 쓰되, 회수한 완료분(done)을 먼저 기록해 재시도로 인한
    # stale(실패) 줄이 남지 않게 한다. done 통계도 함께 재집계해 최종 요약을 정확히.
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in done.values():
            aggregate_and_write(rec, f)

        if workers == 1:
            for global_idx, item in tqdm(pending, desc="평가 중"):
                aggregate_and_write(run_one(global_idx, item), f)
        else:
            # API 응답 대기가 병목 → 스레드 풀로 동시 호출.
            # as_completed 는 메인 스레드에서 순차 소비되므로 집계/쓰기에 락 불필요.
            # (완료 순서대로 기록되어 줄 순서는 global_idx 와 다를 수 있음 — 각 줄에
            #  global_idx 가 있으니 후처리에서 정렬 가능.)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_one, global_idx, item): global_idx
                    for global_idx, item in pending
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="평가 중"):
                    aggregate_and_write(future.result(), f)

    valid = len(indexed) - error_count
    acc = (correct_count / valid * 100) if valid > 0 else 0
    fmt_rate = (format_ok_count / valid * 100) if valid > 0 else 0

    price = PRICING.get(model, {"input": 0, "output": 0})
    cost = (total_input * price["input"] + total_output * price["output"]) / 1_000_000

    print(f"\n=== {model} 결과 ===")
    print(f"정답률: {correct_count}/{valid} = {acc:.1f}%")
    print(f"형식 준수율 (4 step + 정답 라인 모두): {format_ok_count}/{valid} = {fmt_rate:.1f}%")
    if error_count:
        print(f"API 오류: {error_count}건")

    print("\n분야별 정답률:")
    for subset, stats in per_subset.items():
        v = stats["total"] - stats["errors"]
        a = (stats["correct"] / v * 100) if v else 0
        print(f"- {subset}: {stats['correct']}/{v} = {a:.1f}%")

    lang_total = sum(lang_counter.values())
    if lang_total:
        print("\nthink(reasoning_content) 언어 분포:")
        for label, n in lang_counter.most_common():
            print(f"- {label}: {n}/{lang_total} = {n / lang_total * 100:.1f}%")
    else:
        print("\nthink 언어 분포: 기록 없음 (langdetect 미설치 또는 "
              "reasoning_content 부재).")

    print(f"\n입력 토큰: {total_input:,}")
    print(f"출력 토큰: {total_output:,}")
    print(f"예상 비용: ${cost:.4f}")
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
