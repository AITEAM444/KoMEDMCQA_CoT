#!/usr/bin/env python3
"""저장된 eval jsonl 을 통일 파서로 재채점한다 (재생성 X, 저장된 output 만 다시 파싱).

생성은 비싸고 결정론적이라 다시 돌릴 필요 없음 — 채점 기준(파서)만 통일하면
모든 arm 을 동일 잣대로 비교할 수 있다.

통일 파서 우선순위 (단조 보장 — 기존 파서가 답을 낸 건 절대 안 바꾸고,
원래 null 이던 것만 관대층으로 회수한다 → 정확도가 내려갈 수 없음):
  1) strict   : "정답: X" / "정답：X" (괄호 허용)        ← 기존 parse_answer 와 동일
  2) fallback : 마지막 80자에서 가장 뒤의 A-E 한 글자  ← 기존 parse_answer 와 동일
  3) lenient  : "정답은 X" / "답은 X" / "X이다/입니다" / "따라서 X"
                (직후 8자에 부정어(아니/틀리/제외/배제/아님) 오면 거절문맥으로 보고 스킵)
                ↑ 1)·2) 가 모두 실패(=원래 null)일 때만 작동하므로 기존 예측을 덮어쓰지 않음

사용:
  python src/eval/rescore.py <file.jsonl> [<file2.jsonl> ...]
원본은 <file>.bak_strict 로 백업하고 제자리에서 predicted/is_correct 만 갱신한다.
"""
from __future__ import annotations
import json, re, sys, shutil
from pathlib import Path

_STRICT = re.compile(r"정답\s*[::]\s*\(?\s*([A-E])\s*\)?")
_NEG = re.compile(r"^\s*[가-힣]{0,3}?\s*(아니|틀리|제외|배제|아님)")
_LENIENT = [
    re.compile(r"정답\s*[은는이가]?\s*[::]?\s*\(?\s*([A-E])\s*\)?\s*번?"),
    re.compile(r"답\s*[은는이가]?\s*[::]?\s*\(?\s*([A-E])\s*\)?\s*번?"),
    re.compile(r"\(?\s*([A-E])\s*\)?\s*번?\s*(?:이다|입니다)"),
    re.compile(r"따라서\s*[,\s]*\(?\s*([A-E])\s*\)?"),
]


def parse_unified(text: str):
    if not text:
        return None
    # 1) strict — 기존 파서와 동일
    m = list(_STRICT.finditer(text))
    if m:
        return m[-1].group(1)
    # 2) 80자 폴백 — 기존 파서와 동일 (기존 비-null 예측을 그대로 보존)
    for ch in reversed(text.strip()[-80:]):
        if ch in "ABCDE":
            return ch
    # 3) 관대 회수 — 여기 도달했다는 건 원래 null 이었다는 뜻
    best = None  # 거절문맥 제외, 가장 뒤에 등장한 관대 매칭
    for rgx in _LENIENT:
        for mm in rgx.finditer(text):
            if _NEG.search(text[mm.end():mm.end() + 8]):
                continue
            if best is None or mm.start() > best[0]:
                best = (mm.start(), mm.group(1))
    if best:
        return best[1]
    return None


def rescore_file(path: str):
    p = Path(path)
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    old_correct = sum(r["predicted"] == r["gold"] for r in recs)
    old_null = sum(r["predicted"] is None for r in recs)

    changed = 0
    for r in recs:
        new_pred = parse_unified(r.get("output", ""))
        if new_pred != r["predicted"]:
            changed += 1
        r["predicted"] = new_pred
        r["is_correct"] = (new_pred == r["gold"])

    new_correct = sum(r["is_correct"] for r in recs)
    new_null = sum(r["predicted"] is None for r in recs)

    bak = p.with_suffix(p.suffix + ".bak_strict")
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                 encoding="utf-8")

    print(f"[{p.name}]  n={n}")
    print(f"  정답률 {100*old_correct/n:6.2f}% -> {100*new_correct/n:6.2f}%  ({new_correct-old_correct:+d}문항)")
    print(f"  null   {old_null:4d}({100*old_null/n:.2f}%) -> {new_null:4d}({100*new_null/n:.2f}%)")
    print(f"  predicted 변경 {changed}문항   백업={bak.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python src/eval/rescore.py <file.jsonl> [...]")
    for f in sys.argv[1:]:
        rescore_file(f)
