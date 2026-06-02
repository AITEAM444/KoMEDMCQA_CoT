"""C1/C2/C3 가 reject 한 샘플을 한 파일로 모아 비교용으로 정리.
- C1 reject  : 전체(C0) 중 C1 탈락 (표준 baseline 게이트 실패)
- C2 reject  : C1 통과분 중 C2 탈락 (범용 judge 하위)
- C3 reject  : C1 통과분 중 C3 탈락 (반사실 gap 작음 = 사후합리화)
각 레코드에 question/choices/gold/CoT/모든 score 를 함께 담는다.
출력: results/rejected_samples_C1_C2_C3.json
"""
import json

UNIFIED = "data/sft/unified.jsonl"
CF = "data/counterfactual/cf_judged.json"
OUT = "results/rejected_samples_C1_C2_C3.json"


def letter(i):
    return chr(65 + i) if isinstance(i, int) and i >= 0 else None


def main():
    rows = {}
    for line in open(UNIFIED, encoding="utf-8"):
        d = json.loads(line)
        rows[d["id"]] = d

    cf = {r["id"]: r for r in json.load(open(CF, encoding="utf-8"))}

    def cf_signal(i):
        m = (cf.get(i, {}).get("metadata") or {}).get("counterfactual")
        if not m:
            return None, None, None, []
        orig = m.get("orig_score")
        cands = []
        for c in m.get("counterfactuals", []):
            cands.append({
                "target_letter": c.get("target_letter"),
                "cf_score": c.get("cf_score"),
                "cf_cot": c.get("cf_cot"),
            })
        cfs = [c["cf_score"] for c in cands if c["cf_score"] is not None]
        maxcf = max(cfs) if cfs else None
        gap = (orig - maxcf) if (orig is not None and maxcf is not None) else None
        return orig, maxcf, gap, cands

    def build(d):
        i = d["id"]
        f = d.get("filters", {})
        sig = d.get("signals") or {}
        orig, maxcf, gap, cands = cf_signal(i)
        gold = d.get("gold")
        choices = d.get("choices") or []
        rejected_by = []
        if not f.get("C1"):
            rejected_by.append("C1")
        else:  # C1 통과한 것만 C2/C3 reject 후보
            if not f.get("C2"):
                rejected_by.append("C2")
            if not f.get("C3"):
                rejected_by.append("C3")
        return {
            "id": i,
            "subject": d.get("subject"),
            "rejected_by": rejected_by,
            "passed": {"C1": f.get("C1"), "C2": f.get("C2"), "C3": f.get("C3")},
            "question": d.get("question"),
            "choices": {letter(k): c for k, c in enumerate(choices)},
            "gold_letter": letter(gold),
            "gold_text": choices[gold] if isinstance(gold, int) and 0 <= gold < len(choices) else None,
            "teacher_answer": d.get("final_answer"),
            "teacher_correct": (sig.get("c1") or {}).get("correctness"),
            "scores": {
                "c2_score": sig.get("c2_score"),
                "orig_score": orig,
                "max_cf_score": maxcf,
                "gap": gap,
            },
            "c1_gates": sig.get("c1"),
            "original_cot": d.get("think"),
            "counterfactuals": cands,
        }

    c1_rej, c2_rej, c3_rej = [], [], []
    for d in rows.values():
        rec = build(d)
        rb = rec["rejected_by"]
        if "C1" in rb:
            c1_rej.append(rec)
        if "C2" in rb:
            c2_rej.append(rec)
        if "C3" in rb:
            c3_rej.append(rec)

    keyf = lambda r: r["id"]
    c1_rej.sort(key=keyf)
    # C2/C3: gap/score 오름차순(가장 강하게 탈락한 것부터)
    c2_rej.sort(key=lambda r: (r["scores"]["c2_score"] if r["scores"]["c2_score"] is not None else 99))
    c3_rej.sort(key=lambda r: (r["scores"]["gap"] if r["scores"]["gap"] is not None else 99))

    both23 = sorted(set(r["id"] for r in c2_rej) & set(r["id"] for r in c3_rej))

    out = {
        "_meta": {
            "source_unified": UNIFIED,
            "source_cf": CF,
            "definitions": {
                "C1_reject": "C0 중 C1 탈락 (표준 baseline: correctness/readability/format/length)",
                "C2_reject": "C1 통과분 중 C2 탈락 (범용 judge c2_score 하위)",
                "C3_reject": "C1 통과분 중 C3 탈락 (gap=orig_score-max(cf_score) < 1.5)",
            },
            "counts": {
                "C1_reject": len(c1_rej),
                "C2_reject": len(c2_rej),
                "C3_reject": len(c3_rej),
                "C2_and_C3_reject_overlap": len(both23),
                "C2_only": len(c2_rej) - len(both23),
                "C3_only": len(c3_rej) - len(both23),
            },
        },
        "C1_reject": c1_rej,
        "C2_reject": c2_rej,
        "C3_reject": c3_rej,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("counts:", out["_meta"]["counts"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
