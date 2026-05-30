"""
Counterfactual Answer Probing Filter (Inverse MATCHA — 제안 층위)

precompute.py 가 metadata["counterfactual"] 에 채워 둔
  - orig_score        : 원본 CoT 의 "gold 정당화 강도" (1~5)
  - counterfactuals[].cf_score : 각 wrong-answer 정당화 강도
  - counterfactuals[].cf_cot   : R1 이 wrong 답을 정당화하느라 새로 쓴 CoT 본문
를 읽어 두 가지 신호를 계산한다:

판정 신호 (AND — 두 신호가 합의해야 탈락; hedge_gap 비활성 시 gap 단독):
    (1) gap = orig_score - max(cf_score)
        gap < gap_threshold (default 1.5) → 어떤 답이든 비슷한 강도로 정당화됨
        = "답에 무관한 사후합리화" 의심
    (2) hedge_gap = orig_hedge_rate - min(cf_hedge_rate)
        hedge_gap >= hedge_gap_threshold → CF 가 원본만큼 자신 있게 wrong 답 정당화
        = 사후합리화 확증 (해석은 §"Hedge 신호" 절 참조)

    hedge_gap_threshold 가 None(default) 이면 hedge 신호는 metadata 노출만 하고
    필터 결정엔 사용하지 않는다 → gap 단독 판정으로 fallback.

    hedge_gap_threshold 가 설정된 경우 AND 의 의미:
      - gap 단독으로는 over-reject 위험이 있어 (judge 가 3.0 plateau 에 몰리므로),
        hedge_gap 이 사후합리화 패턴을 확증해줄 때만 탈락.
      - hedge_gap << 0 (CF 가 hedge 폭증 = R1 망설임 = robust) 인 경우는
        gap 이 작아도 rescue 됨.

추가 안전망:
    - cf_score 가 모두 None → on_missing 정책
    - orig_score < min_orig_score → 독립 안전망(OR). gap/hedge 판정과 무관하게
      먼저 탈락시킨다. 원본 CoT 가 gold 를 거의 정당화하지 못하는(정당화 실패 수준)
      샘플은 격차 신호를 따질 것 없이 제거한다. min_orig_score=None 이면 비활성.

Hedge 신호 (보조):
  CF CoT 는 R1 이 *틀린 답을 옹호하느라 쓴* 글이다. 추론이 답에 종속적이라면
  R1 은 wrong 답을 정당화하기 어려워 hedge 표지("하지만", "다시 생각", "확실하지"
  등)를 더 많이 쓴다 → cf_hedge_rate > orig_hedge_rate. 사후합리화 CoT 라면
  어떤 답이든 똑같이 자신 있게 쓰므로 cf_hedge_rate ≈ orig_hedge_rate.
  따라서 hedge_gap (orig − cf 최소값) 이 매우 음수에 가깝거나(= CF 가 훨씬
  hedge 많음 = robust) 0 근처(= sophistry) 인 분포를 가진다.

  본 모듈은 hedge 신호를 *항상 계산해서 metadata 에 노출*하되,
  filter 결정엔 기본적으로 사용하지 않는다 (hedge_gap_threshold=None).
  분석/임계값 calibration 후 활성화 가능.

단독 실행: python -m filters.counterfactual.run --input <in.json> --output <out.json>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from filters.base import BaseFilter
from filters.think_final_divergence.run import (
    _KO_HEDGE_PATTERNS,
    _hedge_rate_per_100words,
)
from utils.schema import CoTSample


# R1 이 "다시 생각해보자", "확실하지 않다" 같은 망설임을 *실제로 쓰는 위치는 <think> 블록*
# 이다. 정형 4-step 출력은 이미 결론 정리된 깔끔한 글이라 hedge 가 거의 나오지 않는다.
# 따라서 hedge 신호는 think 블록에서 측정해야 의미가 있다 (formal 출력만 보면 신호 누락).
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_think(text: str) -> str:
    if not text:
        return ""
    m = _THINK_BLOCK_RE.search(text)
    return m.group(1) if m else ""


_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["proposal_filters"]["counterfactual"]


def _compute_hedge_signals(orig_cot: str, cfs: list[dict]) -> dict:
    """원본 CoT 와 각 CF CoT 의 hedge rate 를 계산하고 paired 신호 dict 반환.

    측정 위치는 *<think> 블록 내부* — R1 의 망설임 표현("다시 생각", "확실하지" 등)이
    실제로 나타나는 곳은 think 블록이고, 정형 4-step 출력은 결론 정리 후 깔끔하게 쓰여
    hedge 가 거의 잡히지 않기 때문이다.

    hedge_gap = orig_hedge - min(cf_hedge)
      - 양수 큼: orig 가 hedge 많고 CF 는 자신 있게 wrong 답 정당화 (사후합리화 의심)
      - 0 근처: 양쪽 다 hedge 없음 (judge gap 으로 판단)
      - 음수 큼: CF 가 hedge 폭증 (R1 이 wrong 답을 옹호하느라 망설임 = robust 의심)
    """
    orig_think = _extract_think(orig_cot)
    orig_hedge = _hedge_rate_per_100words(orig_think, _KO_HEDGE_PATTERNS)

    cf_hedge_records = []
    for c in cfs:
        cf_think = _extract_think(c.get("cf_cot") or "")
        cf_hedge = _hedge_rate_per_100words(cf_think, _KO_HEDGE_PATTERNS) if cf_think else None
        cf_hedge_records.append({
            "target_letter": c.get("target_letter"),
            "cf_hedge_rate": round(cf_hedge, 4) if cf_hedge is not None else None,
            "hedge_gap":     round(orig_hedge - cf_hedge, 4) if cf_hedge is not None else None,
        })

    valid_cf_hedges = [r["cf_hedge_rate"] for r in cf_hedge_records if r["cf_hedge_rate"] is not None]
    cf_hedge_min = min(valid_cf_hedges) if valid_cf_hedges else None
    cf_hedge_max = max(valid_cf_hedges) if valid_cf_hedges else None
    cf_hedge_mean = (sum(valid_cf_hedges) / len(valid_cf_hedges)) if valid_cf_hedges else None
    hedge_gap = (orig_hedge - cf_hedge_min) if cf_hedge_min is not None else None
    return {
        "measured_on":     "think_block",
        "orig_hedge_rate": round(orig_hedge, 4),
        "cf_hedge_min":    round(cf_hedge_min, 4) if cf_hedge_min is not None else None,
        "cf_hedge_max":    round(cf_hedge_max, 4) if cf_hedge_max is not None else None,
        "cf_hedge_mean":   round(cf_hedge_mean, 4) if cf_hedge_mean is not None else None,
        "hedge_gap":       round(hedge_gap, 4) if hedge_gap is not None else None,
        "per_cf":          cf_hedge_records,
    }


class CounterfactualFilter(BaseFilter):
    name = "counterfactual"

    def __init__(
        self,
        gap_threshold: float | None = None,
        hedge_gap_threshold: float | None = None,
        min_orig_score: float | None = None,
        on_missing: str | None = None,
    ):
        cfg = _load_config()
        self.gap_threshold = (
            gap_threshold if gap_threshold is not None
            else cfg.get("gap_threshold", 1.5)
        )
        # min_orig_score: 원본 CoT 의 gold 정당화가 약하면(orig_score < 임계값) gap/hedge
        # 와 무관하게 탈락시키는 독립 안전망. None 이면 비활성.
        self.min_orig_score = (
            min_orig_score if min_orig_score is not None
            else cfg.get("min_orig_score", None)
        )
        # hedge_gap_threshold: None(default) 이면 hedge 신호는 metadata 노출만 하고
        # 필터 결정엔 사용하지 않는다. 명시적으로 설정하면 OR 룰에 추가된다.
        self.hedge_gap_threshold = (
            hedge_gap_threshold if hedge_gap_threshold is not None
            else cfg.get("hedge_gap_threshold", None)
        )
        self.on_missing = on_missing or cfg.get("on_missing", "pass")
        if self.on_missing not in {"pass", "fail"}:
            raise ValueError(f"on_missing must be 'pass' or 'fail', got {self.on_missing!r}")

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        meta = (sample.metadata or {}).get("counterfactual")
        if not meta:
            if self.on_missing == "fail":
                return False, None, "counterfactual precompute 결과 없음"
            return True, None, None

        orig_score = meta.get("orig_score")
        cfs = meta.get("counterfactuals", [])
        cf_scores = [c["cf_score"] for c in cfs if c.get("cf_score") is not None]

        if orig_score is None or not cf_scores:
            if self.on_missing == "fail":
                return False, None, "orig_score 또는 cf_scores 누락"
            return True, None, None

        cf_max = max(cf_scores)
        gap = orig_score - cf_max

        # hedge 신호 — judge 와 독립적인 보조 신호. 항상 계산해서 노출.
        hedge_signals = _compute_hedge_signals(sample.cot, cfs)

        sample.metadata.setdefault("counterfactual", {}).update({
            "cf_max":  round(cf_max, 4),
            "gap":     round(gap, 4),
            "hedge":   hedge_signals,
        })

        # 독립 안전망(OR): 원본 CoT 가 gold 를 거의 정당화 못하면 gap/hedge 무관하게 탈락.
        if self.min_orig_score is not None and orig_score < self.min_orig_score:
            return False, gap, (
                f"orig_score={orig_score:.2f}<{self.min_orig_score} "
                f"(원본 CoT 의 gold 정당화 실패 — 안전망)"
            )

        gap_trig = gap < self.gap_threshold
        gap_reason = (
            f"gap={gap:.2f}<{self.gap_threshold} "
            f"(orig={orig_score:.2f}, cf_max={cf_max:.2f})"
        )

        if self.hedge_gap_threshold is None:
            # hedge_gap 비활성 → gap 단독 판정
            if gap_trig:
                return False, gap, gap_reason
            return True, gap, None

        # AND-rule: gap 과 hedge_gap 모두 사후합리화 신호를 가리켜야 탈락
        hg = hedge_signals["hedge_gap"]
        hedge_trig = hg is not None and hg >= self.hedge_gap_threshold
        if gap_trig and hedge_trig:
            hedge_reason = (
                f"hedge_gap={hg:.2f}>={self.hedge_gap_threshold} "
                f"(orig_hedge={hedge_signals['orig_hedge_rate']:.2f}, "
                f"cf_hedge_min={hedge_signals['cf_hedge_min']:.2f}: "
                f"CF가 원본만큼 hedge 적음 = 답 무관 정당화 확증)"
            )
            return False, gap, f"{gap_reason} AND {hedge_reason}"
        return True, gap, None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    CounterfactualFilter().run_from_file(args.input, args.output)
