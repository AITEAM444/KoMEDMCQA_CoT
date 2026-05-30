"""반사실 임계값 — Path 3 검증(확인이지 튜닝 아님). 연구계획 §3/§6.5.

calibrate.py --auto 가 dev gap 분포에서 **비지도 컷**(결정)과 후보들을 내면, 여기서
후보 컷 몇 개만 **dev ACC** 로 찍어 "비지도 픽이 실제 목적함수(ACC)와도 수렴하는지"를
*확인*한다(풀스윕 아님). 세 가지 원칙을 코드로 강제한다:

  ① 확인이지 튜닝 아님 — 비지도 컷을 결정으로 두고 ACC 는 수렴 확인만. ACC-best 로
     컷을 옮기면 그건 (거친) dev 튜닝이므로 analyze 가 그렇게 *명시*해 준다(자동 이동 X).
  ② 분산 체크 — 후보 컷들의 ACC 차이가 seed noise 보다 큰지 본다. 평평하면
     "비지도 컷이 다른 것만큼 좋다"가 정직한 결론(이것도 방어됨).
  ③ 분리도가 검증강도를 조절 — 분리 강하면 골짜기 평평→ACC 둔감(가볍게), 약하면
     컷 민감→ACC 확인이 값짐(seed 늘려). (강도 권고는 calibrate.py --auto 가 출력)

두 모드:
  plan     후보 컷마다 merge-c3(--gap)→export C3→train→**dev** eval 명령을 생성(실행은 무거움).
  analyze  생성된 dev eval 파일들로 mean±std + 분산체크 + 결정(확인/평평/재검토).

사용:
  # 1) 계획 출력 (calibrate --auto 후보를 넣음)
  python src/eval/verify_thresholds.py plan \
      --unified data/unified.jsonl --cf data/cf_judged.json \
      --candidates 1.0 1.5 1.9 --unsup-cut 1.5 --seeds 42 43 --out-sh run_verify.sh
  # 2) (run_verify.sh 실행해 dev eval 생성 후) 분석
  python src/eval/verify_thresholds.py analyze \
      --cut 1.0 results/devacc_g1.0_s*.jsonl --cut 1.5 results/devacc_g1.5_s*.jsonl \
      --cut 1.9 results/devacc_g1.9_s*.jsonl --unsup-cut 1.5
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _tag(g: float) -> str:
    return f"{g:g}".replace(".", "p").replace("-", "m")


# ── plan ────────────────────────────────────────────────────────────────────────
def cmd_plan(args):
    seeds = " ".join(str(s) for s in args.seeds)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             f"# Path 3 임계값 검증 — 후보 {args.candidates} (비지도 컷={args.unsup_cut})",
             "# ⚠ 학습 포함이라 무겁다. 후보 수 × seed 수만큼 LoRA 학습."]
    for g in args.candidates:
        t = _tag(g)
        uni = f"data/_thr/unified_g{t}.jsonl"
        lines += [
            "", f"# ===== gap={g} =====",
            f"python src/dataset/build_arms.py merge-c3 --gap {g} "
            f"--unified {args.unified} --cf {args.cf} --output {uni}",
            f"python src/train/train_lora.py --unified {uni} --arms C3 --seeds {seeds} "
            f"--data-dir data/_thr/sft_g{t} --output-root output/thr_g{t}",
        ]
        for s in args.seeds:
            lines.append(
                f"python src/eval/evaluate.py --model {args.base_model} "
                f"--lora output/thr_g{t}/qwen3-8b-c3-s{s} "
                f"--split {args.dev_split} --output results/devacc_g{t}_s{s}.jsonl"
            )
    # 마지막에 analyze 명령
    analyze = ["", "# ===== 분석 ====="]
    parts = []
    for g in args.candidates:
        parts.append(f"--cut {g} results/devacc_g{_tag(g)}_s*.jsonl")
    analyze.append("python src/eval/verify_thresholds.py analyze "
                   + " ".join(parts) + f" --unsup-cut {args.unsup_cut}")
    lines += analyze

    text = "\n".join(lines)
    print(text)
    if args.out_sh:
        Path(args.out_sh).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_sh).write_text(text + "\n", encoding="utf-8")
        print(f"\n[plan] 저장 → {args.out_sh}  (검토 후 bash 로 실행)")


# ── analyze ──────────────────────────────────────────────────────────────────────
def _acc_of(path: str) -> float | None:
    n = ok = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        ok += int(bool(r.get("is_correct")))
    return (ok / n) if n else None


def cmd_analyze(args):
    # cut → seed accs
    cuts = {}
    for entry in args.cut:
        g = float(entry[0])
        accs = [a for a in (_acc_of(f) for f in entry[1:]) if a is not None]
        if accs:
            cuts[g] = accs
    if not cuts:
        raise SystemExit("[analyze] 유효한 dev eval 파일이 없습니다.")

    summ = {}
    for g, accs in cuts.items():
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) >= 2 else 0.0
        summ[g] = {"mean": mean, "std": std, "n": len(accs), "accs": accs}

    # noise = 컷 내 seed std 의 대표값(있으면 평균, 1 seed뿐이면 0 → 경고)
    stds = [s["std"] for s in summ.values() if s["n"] >= 2]
    noise = statistics.mean(stds) if stds else 0.0
    single_seed = any(s["n"] < 2 for s in summ.values())

    best_g = max(summ, key=lambda g: summ[g]["mean"])
    spread = max(s["mean"] for s in summ.values()) - min(s["mean"] for s in summ.values())
    unsup = args.unsup_cut

    print("\n" + "=" * 60)
    print("임계값 검증 (dev ACC) — 확인이지 튜닝 아님")
    print("=" * 60)
    for g in sorted(summ):
        s = summ[g]
        mark = "  ← 비지도컷" if (unsup is not None and abs(g - unsup) < 1e-9) else ""
        best = "  [ACC-best]" if g == best_g else ""
        print(f"  gap={g:<5g} ACC={s['mean']:.4f} ± {s['std']:.4f} (n={s['n']}){best}{mark}")
    print(f"\n  spread(max-min)={spread:.4f}   seed noise≈{noise:.4f}")
    if single_seed:
        print("  ⚠ 일부 컷이 seed 1개뿐 → 분산 추정 불가. seed≥2 로 재실행 권장(단서 ②).")

    # ── 결정 로직 (자동 이동 금지) ───────────────────────────────────────────────
    verdict = []
    flat = spread <= max(noise, 1e-9)
    unsup_in = unsup is not None and any(abs(g - unsup) < 1e-9 for g in summ)
    if flat:
        verdict.append("📊 ACC 평평(spread ≤ noise) → 그럴듯한 컷 범위에서 ACC 차이가 noise 이하."
                       " '비지도 컷이 다른 것만큼 좋다'가 정직한 결론. 비지도 컷 유지(확인 성공, Path2 충분).")
    elif unsup_in and best_g == unsup:
        verdict.append("✅ 비지도 컷 == ACC-best → 비지도 픽이 실제 목적함수와 수렴. 방어 강함.")
    elif unsup_in:
        d = summ[best_g]["mean"] - summ[unsup]["mean"]
        comb = (summ[best_g]["std"] ** 2 + summ[unsup]["std"] ** 2) ** 0.5
        if d <= comb:
            verdict.append(f"≈ ACC-best(gap={best_g:g})가 비지도컷(gap={unsup:g})보다 +{d:.4f}지만 "
                           f"결합 std({comb:.4f}) 이내 → 사실상 동급. 비지도 컷 유지 가능.")
        else:
            verdict.append(f"⚠ ACC-best(gap={best_g:g})가 비지도컷(gap={unsup:g})보다 분명히 높음(+{d:.4f}>"
                           f"{comb:.4f}=결합std) → 재검토 신호. 컷을 옮기려면 '후보 몇 개에서 "
                           f"dev ACC 로 거친 선택'이라고 포지셔닝을 *명시*하라(비지도→튜닝, 단서 ①).")
    else:
        verdict.append(f"ℹ 비지도 컷({unsup})이 후보에 없음. ACC-best=gap={best_g:g}. "
                       "비지도 컷도 후보에 넣어 비교하라.")
    print("\n결정:")
    for v in verdict:
        print("  " + v)

    if args.output:
        rep = {"summary": {str(g): summ[g] for g in summ}, "noise": noise,
               "spread": spread, "best": best_g, "unsup_cut": unsup,
               "flat": flat, "verdict": verdict}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장 → {args.output}")


def main():
    p = argparse.ArgumentParser(description="반사실 임계값 Path 3 검증 (plan / analyze)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="후보 컷별 train→dev eval 명령 생성")
    pl.add_argument("--unified", required=True, help="train unified jsonl (C0/C1 채워진 것)")
    pl.add_argument("--cf", required=True, help="train cf 채점 파일")
    pl.add_argument("--candidates", nargs="+", type=float, required=True, help="검증할 gap 후보들")
    pl.add_argument("--unsup-cut", type=float, required=True, help="calibrate --auto 가 정한 비지도 컷")
    pl.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    pl.add_argument("--base-model", default="Qwen/Qwen3-8B")
    pl.add_argument("--dev-split", default="dev")
    pl.add_argument("--out-sh", default=None, help="생성 명령을 .sh 로 저장")
    pl.set_defaults(func=cmd_plan)

    an = sub.add_parser("analyze", help="dev eval 파일들로 분산체크 + 결정")
    an.add_argument("--cut", nargs="+", action="append", metavar=("GAP", "FILE"), required=True,
                    help="--cut <gap> <dev_eval1.jsonl> [...]  (여러 번)")
    an.add_argument("--unsup-cut", type=float, default=None, help="비지도 컷(있으면 일치/수렴 판정)")
    an.add_argument("--output", default=None, help="리포트 JSON 저장")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
