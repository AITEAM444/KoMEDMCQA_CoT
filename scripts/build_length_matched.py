"""길이 통제 재실험용 통제군 생성 (학습 trace 길이 교란 ↔ 필터 효과 분리).

- C-randL : C1 풀에서 토큰길이 분포를 C3 에 맞춘 무작위 |C3|개.
            → C3 vs C-randL = 크기+길이 동일, 차이는 '반사실 gap 선별'뿐.
- C-randS : C1 풀에서 토큰길이 분포를 C2(최단)에 맞춘 무작위 |C3|개.
            → C2 vs C-randS = 'C2 우위가 길이(짧음) 때문인가'를 검정.

원본 unified.jsonl 은 건드리지 않고 SFT train 파일만 출력(기존 export 포맷 동일).
"""
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# build_arms.py 와 동일한 export 포맷 (yaml 의존 회피 위해 상수만 인라인)
_L = ["A", "B", "C", "D", "E"]
INSTRUCTION = (
    "다음 한국 의료 자격시험 문제를 풀이하세요. "
    "충분히 추론한 뒤, 마지막 줄에 정확히 \"정답: X\" (X는 A~E 중 하나) 형식으로 답하세요."
)
_TRAIL_ANSWER_RE = re.compile(
    r"(?:\n+|\s)*(?:thus,?\s*|so,?\s*|따라서\s*|그러므로\s*)?정답\s*[::]\s*\(?[A-E]\)?\.?\s*$",
    re.IGNORECASE,
)


def _q_block(question, choices):
    lines = "\n".join(f"{_L[i]}. {c}" for i, c in enumerate(choices))
    return f"[문제]\n{question}\n\n[선택지]\n{lines}"


UNIFIED = ROOT / "data/sft/unified.jsonl"
N_BINS = 20
BUILD_SEED = 42


def load_rows():
    return [json.loads(l) for l in open(UNIFIED, encoding="utf-8") if l.strip()]


def quantile_edges(values, nbins):
    xs = sorted(values)
    return [xs[min(len(xs) - 1, round(i * len(xs) / nbins))] for i in range(nbins + 1)]


def bin_of(x, edges):
    for b in range(len(edges) - 1):
        if x <= edges[b + 1]:
            return b
    return len(edges) - 2


def export(rows, ids, path):
    idset = set(ids)
    out = []
    for r in rows:
        if r["id"] not in idset:
            continue
        think = (r.get("think") or "").strip()
        ans = r.get("final_answer") or _L[r["gold"]]
        if not think or ans is None:
            continue
        think_clean = _TRAIL_ANSWER_RE.sub("", think).strip()
        out.append({
            "messages": [
                {"role": "user", "content": f"{INSTRUCTION}\n\n{_q_block(r['question'], r['choices'])}"},
                {"role": "assistant", "content": f"{think_clean}\n정답: {ans}"},
            ],
            "meta": {"id": r["id"], "subject": r["subject"], "gold": _L[r["gold"]], "answer_used": ans},
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return len(out)


def length_matched(rows, target_flag, name, out_path):
    tok = {r["id"]: r.get("tokens") for r in rows}
    c1_pool = [r["id"] for r in rows if r["filters"].get("C1") and tok.get(r["id"])]
    target = [r["id"] for r in rows if r["filters"].get(target_flag) and tok.get(r["id"])]
    edges = quantile_edges([tok[i] for i in c1_pool], N_BINS)

    pool_by_bin = defaultdict(list)
    for i in c1_pool:
        pool_by_bin[bin_of(tok[i], edges)].append(i)
    tgt_count = defaultdict(int)
    for i in target:
        tgt_count[bin_of(tok[i], edges)] += 1

    rng = random.Random(BUILD_SEED)
    picked = []
    for b, n in tgt_count.items():
        cand = pool_by_bin[b][:]
        rng.shuffle(cand)
        picked.extend(cand[:n])
    n_out = export(rows, picked, out_path)

    tt = [tok[i] for i in target]
    pt = [tok[i] for i in picked]
    print(f"[{name}] target={target_flag}({len(target)})  picked={len(picked)} exported={n_out}")
    print(f"    tokens  target  mean={int(st.mean(tt))} median={int(st.median(tt))}")
    print(f"    tokens  {name:7} mean={int(st.mean(pt))} median={int(st.median(pt))}")
    print(f"    overlap with {target_flag}: {len(set(picked)&set(target))} / {len(picked)}")
    return picked


def main():
    rows = load_rows()
    length_matched(rows, "C3", "C-randL", ROOT / "data/sft/train_C-randL.jsonl")
    length_matched(rows, "C2", "C-randS", ROOT / "data/sft/train_C-randS.jsonl")
    print("\n다음: data/sft/dataset_info.json 에 komed_crandl / komed_crands 등록 후 학습.")


if __name__ == "__main__":
    main()
