"""Build low-data regime SFT slices for the H2 follow-up.

For each requested n, this script exports three same-size training sets:

- C3: top-n C1 examples by counterfactual gap = orig_score - max(cf_score)
- C2: top-n C1 examples by general judge score
- C-rand: random n examples from the C1 pool

It also registers the generated ShareGPT files in data/sft/dataset_info.json and
writes a runnable shell script for LLaMA-Factory training, evaluation, and stats.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = ["A", "B", "C", "D", "E"]
INSTRUCTION = (
    '다음 한국 의료 자격시험 문제를 풀이하세요. 충분히 추론한 뒤, 마지막 줄에 정확히 '
    '"정답: X" (X는 A~E 중 하나) 형식으로 답하세요.'
)
TRAIL_ANSWER_RE = re.compile(
    r"(?:\n+|\s)*(?:thus,?\s*|so,?\s*|따라서\s*|그러므로\s*)?정답\s*[:：]\s*\(?[A-E]\)?\.?\s*$",
    re.IGNORECASE,
)
SHAREGPT_ENTRY = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cf_record_id(sample: dict) -> str | None:
    subset = sample.get("subset")
    if not subset:
        return None
    meta = sample.get("metadata") or {}
    sample_idx = meta.get("sample_idx", sample.get("id"))
    try:
        return f"{subset}_{int(sample_idx):04d}"
    except (TypeError, ValueError):
        return None


def load_gap_by_id(cf_path: Path) -> dict[str, float]:
    data = json.loads(cf_path.read_text(encoding="utf-8"))
    gaps: dict[str, float] = {}
    for sample in data:
        rid = cf_record_id(sample)
        cf = ((sample.get("metadata") or {}).get("counterfactual") or {})
        orig = cf.get("orig_score")
        cfs = cf.get("counterfactuals") or []
        cf_scores = [x.get("cf_score") for x in cfs if x.get("cf_score") is not None]
        if rid is None or orig is None or not cf_scores:
            continue
        gaps[rid] = float(orig) - max(float(x) for x in cf_scores)
    return gaps


def question_block(row: dict) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"]))
    return f"[문제]\n{row['question']}\n\n[선택지]\n{choices}"


def to_sft(row: dict) -> dict:
    think = (row.get("think") or "").strip()
    think = TRAIL_ANSWER_RE.sub("", think).strip()
    ans = row.get("final_answer") or LETTERS[int(row["gold"])]
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{question_block(row)}"},
            {"role": "assistant", "content": f"{think}\n정답: {ans}"},
        ],
        "meta": {
            "id": row["id"],
            "subject": row["subject"],
            "gold": LETTERS[int(row["gold"])],
            "answer_used": ans,
        },
    }


def register_dataset(dataset_info: Path, name: str, data_path: Path, dataset_dir: Path) -> None:
    info = json.loads(dataset_info.read_text(encoding="utf-8")) if dataset_info.exists() else {}
    rel = str(data_path.resolve().relative_to(dataset_dir.resolve()))
    info[name] = {"file_name": rel, **SHAREGPT_ENTRY}
    dataset_info.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def arm_slug(arm: str) -> str:
    return {"C3": "c3", "C2": "c2", "C-rand": "crand"}[arm]


def stratified_counts(pool: list[dict], n: int) -> dict[str, int]:
    counts = Counter(r["subject"] for r in pool)
    raw = {subj: n * count / len(pool) for subj, count in counts.items()}
    out = {subj: int(raw[subj]) for subj in counts}
    remainder = n - sum(out.values())
    order = sorted(counts, key=lambda subj: (raw[subj] - out[subj], counts[subj], subj), reverse=True)
    for subj in order[:remainder]:
        out[subj] += 1
    return out


def take_by_subject(
    ranked: list[dict],
    target_counts: dict[str, int],
    *,
    fill_from_ranked: bool = True,
) -> list[dict]:
    picked: list[dict] = []
    used: set[str] = set()
    current = Counter()
    for row in ranked:
        subj = row["subject"]
        if current[subj] >= target_counts.get(subj, 0):
            continue
        picked.append(row)
        used.add(row["id"])
        current[subj] += 1

    if fill_from_ranked and len(picked) < sum(target_counts.values()):
        for row in ranked:
            if row["id"] in used:
                continue
            picked.append(row)
            used.add(row["id"])
            if len(picked) >= sum(target_counts.values()):
                break
    return picked


def select_rows(
    rows: list[dict],
    gap_by_id: dict[str, float],
    arm: str,
    n: int,
    sample_seed: int,
    stratified: bool,
) -> list[dict]:
    c1_pool = [r for r in rows if r.get("filters", {}).get("C1") is True and (r.get("think") or "").strip()]
    target_counts = stratified_counts(c1_pool, n) if stratified else None
    if arm == "C3":
        pool = [r for r in c1_pool if r["id"] in gap_by_id]
        pool.sort(key=lambda r: (-gap_by_id[r["id"]], str(r["id"])))
        return take_by_subject(pool, target_counts) if target_counts else pool[:n]
    if arm == "C2":
        pool = [r for r in c1_pool if r.get("signals", {}).get("c2_score") is not None]
        pool.sort(key=lambda r: (-float(r["signals"]["c2_score"]), str(r["id"])))
        return take_by_subject(pool, target_counts) if target_counts else pool[:n]
    if arm == "C-rand":
        rng = random.Random(sample_seed)
        if target_counts:
            by_subject: dict[str, list[dict]] = defaultdict(list)
            for row in c1_pool:
                by_subject[row["subject"]].append(row)
            picked = []
            for subj, count in target_counts.items():
                cand = by_subject[subj][:]
                rng.shuffle(cand)
                picked.extend(cand[:count])
            rng.shuffle(picked)
            return picked
        pool = c1_pool[:]
        rng.shuffle(pool)
        return pool[:n]
    raise ValueError(f"unknown arm: {arm}")


def summarize(name: str, rows: list[dict], gap_by_id: dict[str, float]) -> str:
    subj = Counter(r["subject"] for r in rows)
    tokens = [int(r.get("tokens") or 0) for r in rows]
    gaps = [gap_by_id[r["id"]] for r in rows if r["id"] in gap_by_id]
    c2s = [float(r["signals"]["c2_score"]) for r in rows if r.get("signals", {}).get("c2_score") is not None]
    mean_tok = sum(tokens) / max(1, len(tokens))
    parts = [f"{name}: n={len(rows)}, mean_tokens={mean_tok:.1f}, subjects={dict(subj)}"]
    if gaps:
        parts.append(f"gap_mean={sum(gaps) / len(gaps):.3f}")
    if c2s:
        parts.append(f"c2_mean={sum(c2s) / len(c2s):.3f}")
    return ", ".join(parts)


def write_run_script(
    path: Path,
    datasets: list[tuple[str, str, int, int]],
    train_seeds: list[int],
    output_root: str,
    split: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "MODEL=${MODEL:-Qwen/Qwen3-8B}",
        "CONFIG=${CONFIG:-configs/train.yaml}",
        "DATASET_DIR=${DATASET_DIR:-data/sft}",
        f"OUTPUT_ROOT=${{OUTPUT_ROOT:-{output_root}}}",
        "RESULTS=${RESULTS:-results/lowdata}",
        "mkdir -p \"$RESULTS\"",
        "",
    ]
    for arm, dataset, n, sample_seed in datasets:
        slug = arm_slug(arm)
        for train_seed in train_seeds:
            out_dir = f"$OUTPUT_ROOT/qwen3-8b-low-{slug}-n{n}-r{sample_seed}-s{train_seed}"
            result = f"$RESULTS/test_low_{slug}_n{n}_r{sample_seed}_s{train_seed}.jsonl"
            lines += [
                f"echo '[train] {arm} n={n} sample_seed={sample_seed} train_seed={train_seed}'",
                "llamafactory-cli train \"$CONFIG\" "
                f"dataset={dataset} dataset_dir=\"$DATASET_DIR\" output_dir=\"{out_dir}\" "
                f"seed={train_seed} data_seed={train_seed}",
                f"echo '[eval] {arm} n={n} sample_seed={sample_seed} train_seed={train_seed}'",
                "python src/eval/evaluate.py --model \"$MODEL\" "
                f"--lora \"{out_dir}\" --split {split} --output \"{result}\"",
                "",
            ]
    for n in sorted({n for _, _, n, _ in datasets}):
        files = []
        for arm in ["C3", "C-rand", "C2"]:
            slug = arm_slug(arm)
            pattern = f"results/lowdata/test_low_{slug}_n{n}_r*_s*.jsonl"
            files.append(f"--arm {arm} {pattern}")
        lines += [
            f"echo '[stats] n={n}'",
            "python src/eval/stats.py "
            + " ".join(files)
            + f" --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n{n}.json",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build low-data H2 regime datasets.")
    ap.add_argument("--unified", default="data/sft/unified.jsonl")
    ap.add_argument("--cf", default="data/counterfactual/cf_judged.json")
    ap.add_argument("--n", nargs="+", type=int, default=[300, 500, 1000])
    ap.add_argument("--sample-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--train-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--output-root", default="output/lowdata")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--run-script", default="results/run_lowdata_regime.sh")
    ap.add_argument("--summary", default="results/lowdata_design.md")
    ap.add_argument("--no-stratify", action="store_true", help="Do not preserve the C1 subject distribution.")
    args = ap.parse_args()

    unified = ROOT / args.unified
    cf_path = ROOT / args.cf
    data_dir = ROOT / args.data_dir
    dataset_info = data_dir / "dataset_info.json"
    rows = read_jsonl(unified)
    gap_by_id = load_gap_by_id(cf_path)

    datasets: list[tuple[str, str, int, int]] = []
    summary = [
        "# Low-Data Regime Design",
        "",
        f"- unified: `{args.unified}`",
        f"- counterfactual judged: `{args.cf}`",
        f"- n: {args.n}",
        f"- sample seeds: {args.sample_seeds}",
        f"- train seeds: {args.train_seeds}",
        f"- subject stratified: {not args.no_stratify}",
        "",
    ]

    for n in args.n:
        for sample_seed in args.sample_seeds:
            for arm in ["C3", "C2", "C-rand"]:
                picked = select_rows(rows, gap_by_id, arm, n, sample_seed, not args.no_stratify)
                if len(picked) < n:
                    raise SystemExit(f"{arm} n={n} only has {len(picked)} candidates")
                suffix = f"low_{arm_slug(arm)}_n{n}_r{sample_seed}"
                out_path = data_dir / f"train_{suffix}.jsonl"
                write_jsonl(out_path, [to_sft(r) for r in picked])
                dataset = f"komed_{suffix}"
                register_dataset(dataset_info, dataset, out_path, data_dir)
                datasets.append((arm, dataset, n, sample_seed))
                summary.append(f"- `{dataset}` -> `{out_path.relative_to(ROOT)}`")
                summary.append(f"  - {summarize(arm, picked, gap_by_id)}")
            summary.append("")

    write_run_script(ROOT / args.run_script, datasets, args.train_seeds, args.output_root, args.eval_split)
    (ROOT / args.summary).write_text("\n".join(summary), encoding="utf-8")
    print(f"[lowdata] wrote {len(datasets)} datasets")
    print(f"[lowdata] run script: {args.run_script}")
    print(f"[lowdata] summary: {args.summary}")


if __name__ == "__main__":
    main()
