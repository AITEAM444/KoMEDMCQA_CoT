"""리포트 생성 — arm 크기 + 신호 분포 + 시드 통계를 마크다운으로 종합. 연구계획 §6.

두 입력을 모아 사람이 읽는 보고서(Markdown)를 만든다:
  · unified.jsonl  (build_arms 산출) → arm 크기표 + 반사실 신호(gap/orig_score) 분포 +
                                       C1∩¬C3(H1 분석 대상) 건수
  · stats.json     (stats.py 산출, 선택) → arm ACC mean±std/CI + McNemar 표

사용:
    python scripts/make_report.py \
        --unified data/unified.jsonl \
        --stats results/stats.json \
        --output results/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ["C0", "C1", "C2", "C3", "C-rand"]


def _read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if not xs:
        return {q: None for q in qs}
    s = sorted(xs)
    out = {}
    for q in qs:
        idx = min(int(q * (len(s) - 1) + 0.5), len(s) - 1)
        out[q] = round(s[idx], 3)
    return out


def section_arms(rows) -> list[str]:
    L = ["## Arm 크기", "", "| Arm | 전체 | " + " | ".join(s.capitalize() for s in
         sorted({r["subject"] for r in rows})) + " |",
         "|---|---|" + "---|" * len({r["subject"] for r in rows})]
    subs = sorted({r["subject"] for r in rows})
    for arm in ARMS:
        n = sum(1 for r in rows if r.get("filters", {}).get(arm) is True)
        per = []
        for sub in subs:
            per.append(str(sum(1 for r in rows
                               if r.get("subject") == sub and r.get("filters", {}).get(arm) is True)))
        L.append(f"| {arm} | {n} | " + " | ".join(per) + " |")
    L.append("")
    # H1 대상: C1 통과 ∩ C3 탈락
    h1 = sum(1 for r in rows
             if r.get("filters", {}).get("C1") is True and r.get("filters", {}).get("C3") is False)
    L += [f"**H1 분석 대상** (C1 통과 ∩ C3 탈락): **{h1}건** — judge 가 못 잡는 사후합리화 후보.", ""]
    return L


def section_signals(rows) -> list[str]:
    """반사실 신호 분포 — signals.cf 또는 metadata 어디에 있든 gap/orig_score 수집."""
    gaps, origs = [], []
    for r in rows:
        sig = r.get("signals", {}) or {}
        cf = sig.get("counterfactual") or sig.get("cf") or {}
        g = cf.get("gap")
        o = cf.get("orig_score")
        if g is not None:
            gaps.append(g)
        if o is not None:
            origs.append(o)
    c2 = [r["signals"]["c2_score"] for r in rows
          if (r.get("signals", {}) or {}).get("c2_score") is not None]
    L = ["## 신호 분포", ""]
    if not gaps and not origs and not c2:
        L += ["_신호 없음 (merge-c3 / judge_general 미실행, 또는 unified 에 signals 미포함)._", ""]
        return L
    if gaps:
        q = _quantiles(gaps)
        L += [f"- **gap** (orig−max cf): n={len(gaps)}, 평균={statistics.mean(gaps):.3f}, "
              f"min/Q1/med/Q3/max = {q[0.0]}/{q[0.25]}/{q[0.5]}/{q[0.75]}/{q[1.0]}"]
    if origs:
        q = _quantiles(origs)
        L += [f"- **orig_score**: n={len(origs)}, 평균={statistics.mean(origs):.3f}, "
              f"min/Q1/med/Q3/max = {q[0.0]}/{q[0.25]}/{q[0.5]}/{q[0.75]}/{q[1.0]}"]
    if c2:
        q = _quantiles(c2)
        L += [f"- **c2_score** (범용 judge): n={len(c2)}, 평균={statistics.mean(c2):.3f}, "
              f"min/Q1/med/Q3/max = {q[0.0]}/{q[0.25]}/{q[0.5]}/{q[0.75]}/{q[1.0]} "
              f"— 분포가 상단에 몰리면(plateau) H3 의 '일반 judge 변별 상실' 신호."]
    L.append("")
    return L


def section_stats(stats: dict) -> list[str]:
    L = ["## 다운스트림 정확도 (seed mean ± std, 95% CI)", "",
         "| Arm | mean | std | CI95 | seeds | n |", "|---|---|---|---|---|---|"]
    for name, a in stats.get("arms", {}).items():
        ci = a.get("ci95", [None, None])
        L.append(f"| {name} | {a.get('mean')} | {a.get('std')} | "
                 f"[{ci[0]}, {ci[1]}] | {a.get('n_seeds')} | {a.get('n_items_common')} |")
    L.append("")
    mc = stats.get("mcnemar", [])
    if mc:
        L += ["### McNemar 짝지은 검정", "",
              "| 비교 | A만정답 | B만정답 | 불일치 | p | 판정 |", "|---|---|---|---|---|---|"]
        for m in mc:
            sig = "유의" if m.get("p_value", 1) < 0.05 else "비유의"
            L.append(f"| {m.get('pair')} | {m.get('A_only_correct')} | {m.get('B_only_correct')} | "
                     f"{m.get('discordant')} | {m.get('p_value')} | {sig} |")
        L.append("")
    # 가설 요약(존재하는 arm 만)
    arms = stats.get("arms", {})
    def acc(n): return arms.get(n, {}).get("mean")
    L += ["### 가설 점검 (참고)", ""]
    if acc("C3") is not None and acc("C1") is not None:
        L.append(f"- H2 (C3 > C1): {acc('C3')} vs {acc('C1')} → "
                 f"{'✅' if acc('C3') > acc('C1') else '❌'}")
    if acc("C3") is not None and acc("C-rand") is not None:
        L.append(f"- H2 통제 (C3 > C-rand): {acc('C3')} vs {acc('C-rand')} → "
                 f"{'✅' if acc('C3') > acc('C-rand') else '❌'}")
    if acc("C3") is not None and acc("C2") is not None:
        L.append(f"- H3 (C3 > C2, 동일 크기): {acc('C3')} vs {acc('C2')} → "
                 f"{'✅' if acc('C3') > acc('C2') else '❌'}")
    L.append("")
    return L


def main():
    p = argparse.ArgumentParser(description="arm 크기/신호/통계 → 마크다운 리포트")
    p.add_argument("--unified", help="build_arms unified jsonl (arm 크기·신호)")
    p.add_argument("--stats", help="stats.py 산출 JSON (ACC/McNemar)")
    p.add_argument("--output", default="results/report.md")
    args = p.parse_args()

    if not args.unified and not args.stats:
        raise SystemExit("--unified 또는 --stats 중 최소 하나가 필요합니다.")

    L = ["# KoMED 실험 리포트", ""]
    if args.unified:
        rows = _read_jsonl(args.unified)
        L += [f"_unified: `{args.unified}` ({len(rows)}건)_", ""]
        L += section_arms(rows)
        L += section_signals(rows)
    if args.stats:
        stats = json.load(open(args.stats, encoding="utf-8"))
        L += [f"_stats: `{args.stats}`_", ""]
        L += section_stats(stats)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(L), encoding="utf-8")
    print(f"리포트 저장 → {args.output}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
