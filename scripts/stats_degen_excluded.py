"""degen(무응답 None + 루프 >=6000자) 제외 후 stats.py 와 동일 방식으로
   seed mean±std + 부트스트랩 95% CI + McNemar 재계산.

stats.py 의 함수를 그대로 import 해 일관성 유지. 차이는 입력 로더뿐:
각 arm·seed 에서 degenerate item 의 key 를 아예 제거(=오답 아님, 모집단에서 빠짐).
→ McNemar 공통 item = '두 arm 모두 모든 seed 에서 멀쩡히 답한 문항'.
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
import stats as S  # noqa: E402

ARMS = ["c0", "c1", "c2", "c3", "c-rand"]
LOOP_CHARS = 6000
N_BOOT = 10000
ALPHA = 0.05


def is_degen(r):
    return r.get("predicted") is None or len(r.get("output") or "") >= LOOP_CHARS


def load_eval_clean(path):
    """{(subset, idx): is_correct} 단, degenerate 는 key 제거."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if is_degen(r):
            continue
        out[(r["subset"], int(r["sample_idx"]))] = int(bool(r.get("is_correct")))
    return out


def summarize(name, seed_dicts, rng):
    seed_dicts = [d for d in seed_dicts if d]
    seed_accs = [statistics.mean(d.values()) for d in seed_dicts]
    # McNemar 용: 3시드 모두 멀쩡한 공통 item (짝 비교의 정합성)
    per_item_common = S._seed_avg_per_item(seed_dicts)
    # CI 용: 합집합 — 각 item 을 '멀쩡했던 seed' 들에서만 평균(전3시드 강제 X).
    #        공통집합(쉬운 문항 편중)이 CI 를 위로 끌어올리는 편향 제거.
    union = defaultdict(list)
    for d in seed_dicts:
        for k, v in d.items():
            union[k].append(v)
    per_item_union = {k: statistics.mean(vs) for k, vs in union.items()}
    lo, hi = S._bootstrap_ci(list(per_item_union.values()), N_BOOT, ALPHA, rng)
    return {
        "arm": name,
        "seed_accuracies": [round(a, 4) for a in seed_accs],
        "mean": round(statistics.mean(seed_accs), 4),
        "std": round(statistics.stdev(seed_accs), 4) if len(seed_accs) >= 2 else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_items_union": len(per_item_union),
        "n_items_common": len(per_item_common),
        "_per_item": per_item_common,
    }


def main():
    rng = random.Random(42)
    arms = {}
    for a in ARMS:
        seed_dicts = [load_eval_clean(ROOT / f"results/test_{a}_s{s}.jsonl") for s in (42, 43, 44)]
        arms[a] = summarize(a, seed_dicts, rng)

    print("=" * 70)
    print(f"degen 제외(None + >={LOOP_CHARS}자) — seed mean±std, 95% bootstrap CI")
    print("=" * 70)
    for a in ARMS:
        x = arms[a]
        print(f"  {a:7s}: {x['mean']:.4f} ± {x['std']:.4f}  "
              f"CI95[{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]  "
              f"n합집합={x['n_items_union']} n공통={x['n_items_common']}  seeds={x['seed_accuracies']}")

    pairs = [("c3", "c1"), ("c3", "c-rand"), ("c3", "c2"),
             ("c2", "c1"), ("c2", "c-rand"), ("c1", "c-rand")]
    print("\n" + "=" * 70)
    print("McNemar (degen 제외, 두 arm 모두 멀쩡히 답한 공통 item, seed-과반 이진화)")
    print("=" * 70)
    mc = []
    for an, bn in pairs:
        m = S.mcnemar(arms[an]["_per_item"], arms[bn]["_per_item"])
        m["pair"] = f"{an} vs {bn}"
        mc.append(m)
        sig = "유의(p<0.05)" if m["p_value"] < 0.05 else "비유의"
        print(f"  {an:6} vs {bn:6}: {an}만정답={m['A_only_correct']:4} {bn}만정답={m['B_only_correct']:4} "
              f"불일치={m['discordant']:4} n공통={m['n_common']:4} p={m['p_value']:.4g} [{m['method']}] → {sig}")

    out = {
        "config": {"loop_chars": LOOP_CHARS, "bootstrap": N_BOOT, "alpha": ALPHA},
        "arms": {a: {k: v for k, v in arms[a].items() if k != "_per_item"} for a in ARMS},
        "mcnemar": mc,
    }
    Path(ROOT / "results/stats_degen_excluded.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved → results/stats_degen_excluded.json")


if __name__ == "__main__":
    main()
