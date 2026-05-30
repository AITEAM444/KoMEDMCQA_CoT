"""평가 통계 — 시드 평균±std + 부트스트랩 CI + McNemar 짝지은 검정.

연구계획 §6.5: 단일 run 의 1~2% ACC 차이는 노이즈. arm 당 ≥3 seed 로 돌려
mean±std + 95% CI 로 보고하고, arm 간 차이는 짝지은 검정(McNemar)으로 본다.

입력: evaluate.py 가 만든 per-arm·per-seed 결과 jsonl
      (각 줄 = {subset, sample_idx, gold, predicted, is_correct, ...}).

계산:
  · arm 별 seed 정확도 리스트 → mean ± std(ddof=1)
  · 부트스트랩 95% CI : seed-평균 per-item 정확도 벡터에서 item 재표집 (item-level)
  · McNemar : 두 arm 의 *공통 item* 에서 seed-과반 이진화 후 불일치쌍(b,c)으로 검정
              (b+c<25 → 정확 이항검정, 아니면 연속성 보정 χ² 정규근사)

외부 의존 없음(scipy 불필요) — math/random 만 사용.

사용:
    python src/eval/stats.py \
        --arm C1 results/eval_C1_s1.jsonl results/eval_C1_s2.jsonl results/eval_C1_s3.jsonl \
        --arm C3 results/eval_C3_s1.jsonl results/eval_C3_s2.jsonl results/eval_C3_s3.jsonl \
        --mcnemar C3 C1 --mcnemar C3 C-rand \
        --output results/stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _load_eval(path: str) -> dict[tuple[str, int], int]:
    """평가 jsonl → {(subset, sample_idx): is_correct(0/1)}."""
    out: dict[tuple[str, int], int] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            key = (r["subset"], int(r["sample_idx"]))
            out[key] = int(bool(r.get("is_correct")))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return out


def _seed_avg_per_item(seed_dicts: list[dict]) -> dict[tuple[str, int], float]:
    """seed 들의 공통 item 에 대해 per-item 평균 정확도(0~1)."""
    if not seed_dicts:
        return {}
    common = set(seed_dicts[0])
    for d in seed_dicts[1:]:
        common &= set(d)
    return {k: statistics.mean(d[k] for d in seed_dicts) for k in common}


def _bootstrap_ci(values: list[float], n_boot: int, alpha: float, rng: random.Random
                  ) -> tuple[float, float]:
    """item-level 부트스트랩 백분위 CI (평균의 분포)."""
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def _norm_sf(z: float) -> float:
    """표준정규 생존함수 P(Z>z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _binom_two_sided_p(b: int, c: int) -> float:
    """정확 이항검정 (p=0.5) 양측 — 불일치쌍이 적을 때(McNemar exact)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 양측 = 2 * P(X <= k), X~Bin(n, 0.5), 1.0 로 클립
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a: dict[tuple[str, int], float], b: dict[tuple[str, int], float]) -> dict:
    """두 arm 의 공통 item 에서 seed-과반 이진화 후 McNemar 검정.

    b_ = A 맞고 B 틀림, c_ = A 틀리고 B 맞음. discordant = b_+c_.
    """
    common = set(a) & set(b)
    ba = cb = both = neither = 0
    for k in common:
        av = a[k] >= 0.5
        bv = b[k] >= 0.5
        if av and not bv:
            ba += 1
        elif bv and not av:
            cb += 1
        elif av and bv:
            both += 1
        else:
            neither += 1
    disc = ba + cb
    if disc == 0:
        p, method, stat = 1.0, "no_discordant", 0.0
    elif disc < 25:
        p, method = _binom_two_sided_p(ba, cb), "exact_binomial"
        stat = float(min(ba, cb))
    else:
        stat = (abs(ba - cb) - 1) ** 2 / disc          # 연속성 보정 χ²(df=1)
        p, method = math.erfc(math.sqrt(stat / 2)), "chi2_continuity"  # χ²(1) 우측꼬리 = 2*Φ̄(√stat)
    return {
        "n_common": len(common),
        "A_only_correct": ba, "B_only_correct": cb,
        "both_correct": both, "neither_correct": neither,
        "discordant": disc, "statistic": round(stat, 4),
        "p_value": round(p, 6), "method": method,
    }


def summarize_arm(name: str, paths: list[str], n_boot: int, alpha: float, rng: random.Random
                  ) -> dict:
    seed_dicts = [_load_eval(p) for p in paths]
    seed_dicts = [d for d in seed_dicts if d]
    if not seed_dicts:
        raise SystemExit(f"[stats] arm {name}: 유효한 결과 파일이 없습니다 ({paths})")
    seed_accs = [statistics.mean(d.values()) for d in seed_dicts]
    per_item = _seed_avg_per_item(seed_dicts)
    ci_lo, ci_hi = _bootstrap_ci(list(per_item.values()), n_boot, alpha, rng)
    mean = statistics.mean(seed_accs)
    std = statistics.stdev(seed_accs) if len(seed_accs) >= 2 else 0.0

    # 과목별 (seed-평균 per-item 기준)
    by_sub: dict[str, list[float]] = defaultdict(list)
    for (sub, _idx), v in per_item.items():
        by_sub[sub].append(v)
    per_subject = {sub: {"acc": round(statistics.mean(vs), 4), "n": len(vs)}
                   for sub, vs in sorted(by_sub.items())}

    return {
        "arm": name, "n_seeds": len(seed_accs), "n_items_common": len(per_item),
        "seed_accuracies": [round(a, 4) for a in seed_accs],
        "mean": round(mean, 4), "std": round(std, 4),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "per_subject": per_subject,
        "_per_item": per_item,   # McNemar 용(리포트에서 제거)
    }


def main():
    p = argparse.ArgumentParser(description="arm 별 ACC mean±std + 부트스트랩 CI + McNemar")
    p.add_argument("--arm", nargs="+", action="append", metavar=("NAME", "FILE"),
                   required=True, help="--arm <이름> <seed1.jsonl> [seed2 ...] (여러 번 반복 가능)")
    p.add_argument("--mcnemar", nargs=2, action="append", metavar=("ARM_A", "ARM_B"),
                   default=[], help="두 arm 간 McNemar (여러 번 반복 가능)")
    p.add_argument("--bootstrap", type=int, default=10000, help="부트스트랩 반복(default 10000)")
    p.add_argument("--alpha", type=float, default=0.05, help="CI 유의수준(default 0.05 → 95%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="리포트 JSON 저장 경로")
    args = p.parse_args()

    rng = random.Random(args.seed)
    arms: dict[str, dict] = {}
    for entry in args.arm:
        name, paths = entry[0], entry[1:]
        if not paths:
            raise SystemExit(f"[stats] arm {name}: 결과 파일 경로가 필요합니다.")
        arms[name] = summarize_arm(name, paths, args.bootstrap, args.alpha, rng)

    # ── 콘솔: arm 요약 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ARM 별 정확도 (seed mean ± std, 95% bootstrap CI)")
    print("=" * 64)
    for name, a in arms.items():
        print(f"  {name:8s}: {a['mean']:.4f} ± {a['std']:.4f}  "
              f"CI95[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]  "
              f"(seeds={a['n_seeds']}, n={a['n_items_common']})")
        print(f"           seed accs={a['seed_accuracies']}")

    # ── McNemar ────────────────────────────────────────────────────────────────
    mc_results = []
    if args.mcnemar:
        print("\n" + "=" * 64)
        print("McNemar 짝지은 검정 (seed-과반 이진화, 공통 item)")
        print("=" * 64)
    for an, bn in args.mcnemar:
        if an not in arms or bn not in arms:
            print(f"  ⚠ {an} vs {bn}: arm 미정의 — 건너뜀")
            continue
        m = mcnemar(arms[an]["_per_item"], arms[bn]["_per_item"])
        m["pair"] = f"{an} vs {bn}"
        mc_results.append(m)
        sig = "유의(p<0.05)" if m["p_value"] < 0.05 else "비유의"
        print(f"  {an} vs {bn}: {an}만정답={m['A_only_correct']}  {bn}만정답={m['B_only_correct']}  "
              f"불일치={m['discordant']}  p={m['p_value']:.4g} [{m['method']}] → {sig}")

    # ── 저장 ────────────────────────────────────────────────────────────────────
    if args.output:
        report = {
            "arms": {n: {k: v for k, v in a.items() if k != "_per_item"} for n, a in arms.items()},
            "mcnemar": mc_results,
            "config": {"bootstrap": args.bootstrap, "alpha": args.alpha, "seed": args.seed},
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n리포트 저장 → {args.output}")


if __name__ == "__main__":
    main()
