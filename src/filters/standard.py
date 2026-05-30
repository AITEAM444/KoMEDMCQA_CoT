"""C1 — 표준 baseline 필터 ①~④ (DeepSeek-R1 2501.12948 정제 절차, 고정값).

새 설계상 학습 타겟은 reasoning_content(내부추론)이므로 4-Step 형식이 아니라
**reasoning(think) 기준**으로 판정한다. 입력은 generate_traces.py 출력 레코드 dict.

  ① correctness  : Teacher 최종답(predicted) == gold
  ② readability  : 깨진 혼합언어/코드블록/run-on 제거 (영어 비중은 무시 — §3② 한글비율 비채택)
  ③ format_parse : 답 파싱 가능 + reasoning 존재 (형식/파싱 게이트)
  ④ length       : reasoning 길이가 [min, max] (너무 짧음=추론 없음 / 너무 김=반복·발산)

config(configs/pipeline_config.yaml: c1_filters)에서 임계값을 읽되, 없으면 기본값.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline_config.yaml"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"정답\s*[::]\s*\(?([A-E])\)?")
_CODEBLOCK_RE = re.compile(r"```")
# 정상 텍스트 문자: 한글, 영문, 숫자, 공백, 일반 문장부호·기호. 그 외(他 CJK·제어·이상기호)는 'garbage'.
_OK_CHAR_RE = re.compile(
    r"[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,;:!?()\[\]{}<>/+\-*=%·…“”‘’\"'`~@#&^|\\$_°±×÷→←↑↓°℃μ㎎㎍㎖㎗ℓ]"
)


def _load_cfg() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("c1_filters", {}) or {}
    except FileNotFoundError:
        return {}


def _think(rec: dict) -> str:
    """판정 대상 추론 텍스트. reasoning_content 우선, 없으면 reasoning/cot 의 <think> 추출."""
    rc = (rec.get("reasoning_content") or "").strip()
    if rc:
        return rc
    body = rec.get("reasoning") or rec.get("cot") or ""
    m = _THINK_RE.search(body)
    return (m.group(1) if m else body).strip()


def _to_letter(v):
    if isinstance(v, str):
        v = v.strip().upper()
        return v if v in ("A", "B", "C", "D", "E") else None
    if isinstance(v, int):
        return "ABCDE"[v - 1] if 1 <= v <= 5 else ("ABCDE"[v] if 0 <= v < 5 else None)
    return None


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = len(_OK_CHAR_RE.sub("", text))
    return bad / max(len(text), 1)


def _max_run_no_space(text: str) -> int:
    return max((len(t) for t in text.split()), default=0)


def check_correctness(rec, gold_letter):
    pred = _to_letter(rec.get("predicted"))
    return pred is not None and pred == gold_letter


def check_format_parse(rec, gold_letter):
    """답 파싱 가능 + 추론 존재."""
    pred = _to_letter(rec.get("predicted"))
    return pred is not None and bool(_think(rec))


def check_readability(rec, cfg):
    """깨진 혼합언어/코드블록/run-on 제거. 영어 비중 자체는 보지 않음."""
    t = _think(rec)
    if not t:
        return False
    if _CODEBLOCK_RE.search(t):
        return False
    if _garbage_ratio(t) > cfg.get("garbage_ratio_max", 0.15):
        return False
    if _max_run_no_space(t) > cfg.get("max_run_no_space", 200):
        return False
    return True


def check_length(rec, cfg):
    t = _think(rec)
    n = len(t.split())
    return cfg.get("min_words", 20) <= n <= cfg.get("max_words", 4000)


def passes_c1(rec, gold_letter, cfg=None):
    """C1 ①~④ 전부 통과 여부 + 개별 결과 dict."""
    cfg = cfg if cfg is not None else _load_cfg()
    results = {
        "correctness": check_correctness(rec, gold_letter),
        "readability": check_readability(rec, cfg),
        "format_parse": check_format_parse(rec, gold_letter),
        "length": check_length(rec, cfg),
    }
    return all(results.values()), results
