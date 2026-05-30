"""
Judge 동등성 비교 — deepseek-chat vs solar-pro3 (Counterfactual judge 채택 검증).

목적:
    Counterfactual(Inverse-MATCHA) judge 를 deepseek-chat 에서 solar-pro3 로
    교체해도 동일하게 작동하는지 검증한다. 통과하면 solar-pro3 를 Judge 모델로 채택.

원칙(공정성):
    두 judge 가 *완전히 동일한 CoT* 를 채점해야 한다. 따라서 cf_cot 는 새로
    생성하지 않고, 이미 deepseek-reasoner 로 생성되어 precompute 결과(metadata
    ["counterfactual"].counterfactuals[].cf_cot)에 박혀 있는 것을 재사용한다.
    orig CoT 는 sample.cot 를 그대로 쓴다.

    각 judging 인스턴스(원본 1 + CF k 개)마다 두 judge 를 호출해
    {supported_letter, matches_target, score} 를 수집하고, keep/reject 는 실제
    CounterfactualFilter.filter_sample 로직을 그대로 재사용해 judge 별로 산출한다.

지표(채택 기준):
    1. supported_letter 일치율  >= 80%  (또는 Cohen's kappa >= 0.6)
    2. matches_target 일치율    >= 85%
    3. score Spearman rho       >= 0.65
    4. keep/reject 결정 일치율  >= 85%   (전체 샘플 중 같은 reject 집합)
    5. id4 sanity               : 두 judge 모두 supported_letter == "A" (원본 채점)

전제:
    UPSTAGE_API_KEY  (solar-pro3),  DEEPSEEK_API_KEY (deepseek-chat)
    --input 은 cf_cot 가 담긴 precompute 출력 JSON (필터링 전 전체 샘플).

사용:
    python scripts/compare_judges.py \
        --input data/filtered/counterfactual_output_precomputed.json \
        --output results/judge_compare/solar_vs_deepseek.json \
        --judge-reps 1 --workers 8
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (filters/, utils/)

from filters.counterfactual.judge import (
    _CF_RUBRIC,
    _LETTERS,
    _call_judge_with_model,
    _extract_internal_reasoning,
    _format_choices,
)
from filters.counterfactual.run import CounterfactualFilter
from filters.judge.score_faithfulness import parse_geval_response
from utils.data_loader import load_samples


# ── judge backends ────────────────────────────────────────────────────────────
# 기본 백엔드: mindlogic API Gateway (OpenAI 호환). 단일 키로 ref/cand 둘 다 호출.
DEFAULT_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
DEFAULT_KEY_ENV = "GATEWAY_API_KEY"


def _make_client(base_url: str, key_env: str):
    from openai import OpenAI
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} 환경변수가 필요합니다 (judge 백엔드).")
    if base_url in (None, "", "openai"):   # OpenAI 직결 (api.openai.com)
        return OpenAI(api_key=key)
    return OpenAI(api_key=key, base_url=base_url)


# reasoning 모델(gpt-5/o-시리즈)은 temperature 미지원 + max_completion_tokens 사용 + thinking 토큰 소비
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _judge_call(client, model: str, prompt: str, temperature: float, max_tokens: int = 1024) -> str:
    """모델별 파라미터 차이를 흡수. reasoning 모델은 temperature 생략 + max_completion_tokens(여유)."""
    if _is_reasoning(model):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max(max_tokens, 4000),   # thinking 토큰 여유 확보
        )
        return resp.choices[0].message.content
    return _call_judge_with_model(client, model, prompt, temperature, max_tokens)


def _mode(values: list):
    """최빈값. 동률이면 첫 등장 순. 빈 리스트는 None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def judge_one(
    client, model: str, question: str, cot: str, target_idx: int, choices,
    *, reps: int, temperature: float, max_chars: int = 8000, sleep_on_error: float = 2.0,
) -> dict:
    """단일 (cot, target) 을 채점해 {supported_letter, matches_target, score} 반환.

    judge.py 의 score_convincingness 와 동일한 프롬프트·파서를 재사용하되,
    score 만 버리지 않고 supported_letter / matches_target 까지 보존한다.
    reps>1 이면 score=평균, supported_letter=최빈, matches_target=과반.
    """
    if not (0 <= target_idx < len(_LETTERS)):
        return {"supported_letter": None, "matches_target": None, "score": None, "n_ok": 0}

    target_letter = _LETTERS[target_idx]
    internal_cot = _extract_internal_reasoning((cot or "")[:max_chars])
    prompt = _CF_RUBRIC.format(
        target_letter=target_letter,
        question=(question or "").strip(),
        choices=_format_choices(choices),
        cot=internal_cot,
    )

    scores, letters, matches = [], [], []
    n_api_err = 0          # 예외(연결/인증/쿼터 등) 발생 횟수 — 파싱실패와 구분
    last_err = None
    for _ in range(max(reps, 1)):
        parsed = None
        for _ in range(3):
            try:
                text = _judge_call(client, model, prompt, temperature)
                parsed = parse_geval_response(text)
                if parsed is not None:
                    break
            except Exception as e:
                n_api_err += 1
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if sleep_on_error:
                time.sleep(sleep_on_error)
        if parsed is not None:
            try:
                scores.append(int(parsed["score"]))
            except (KeyError, TypeError, ValueError):
                pass
            letters.append(parsed.get("supported_letter"))
            mt = parsed.get("matches_target")
            if isinstance(mt, bool):
                matches.append(mt)

    return {
        "supported_letter": _mode(letters),
        "matches_target": _mode(matches),
        "score": (sum(scores) / len(scores)) if scores else None,
        "n_ok": len(scores),
        "n_api_err": n_api_err,      # >0 이고 n_ok==0 이면 API 실패(토큰소진 등)
        "last_err": last_err,
    }


def judge_sample(client, model: str, sample, cfs: list[dict], *, reps: int, temp: float) -> dict:
    """원본(target=gold) + 각 CF(target=wrong) 채점 결과를 모은다."""
    orig = judge_one(client, model, sample.question, sample.cot, sample.answer,
                     sample.choices, reps=reps, temperature=temp)
    cf_out = []
    for c in cfs:
        t_idx = c["target_idx"]
        res = judge_one(client, model, sample.question, c.get("cf_cot") or "", t_idx,
                        sample.choices, reps=reps, temperature=temp)
        cf_out.append({"target_idx": t_idx, "target_letter": c.get("target_letter"),
                       "cf_cot": c.get("cf_cot"), **res})
    # 이 샘플이 전부 API 실패(채점 0건 + 예외 발생)인지 — 토큰소진 감지용
    calls = [orig] + cf_out
    total_ok = sum(c.get("n_ok", 0) for c in calls)
    total_err = sum(c.get("n_api_err", 0) for c in calls)
    api_dead = (total_ok == 0 and total_err > 0)
    return {"orig": orig, "cfs": cf_out, "api_dead": api_dead,
            "last_err": next((c.get("last_err") for c in calls if c.get("last_err")), None)}


def decide_keep(sample, judged: dict, k: int) -> bool:
    """judge 결과로 metadata 를 구성하고 실제 CounterfactualFilter 로 keep/reject 판정."""
    meta_cf = {
        "k": k,
        "orig_target_letter": _LETTERS[sample.answer],
        "orig_score": judged["orig"]["score"],
        "counterfactuals": [
            {"target_idx": c["target_idx"], "target_letter": c["target_letter"],
             "cf_cot": c["cf_cot"], "cf_score": c["score"], "error": None}
            for c in judged["cfs"]
        ],
    }
    s = copy.copy(sample)
    s.metadata = {**(sample.metadata or {}), "counterfactual": meta_cf}
    passed, _, _ = CounterfactualFilter().filter_sample(s)
    return passed


def cohen_kappa(a: list, b: list) -> float | None:
    """범주형 라벨 두 리스트의 Cohen's kappa."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    cats = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    po = sum(1 for x, y in pairs if x == y) / n
    ax = Counter(x for x, _ in pairs)
    bx = Counter(y for _, y in pairs)
    pe = sum((ax[c] / n) * (bx[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    p = argparse.ArgumentParser(description="기준(ref) judge vs 후보(cand) judge 동등성 비교")
    p.add_argument("--input", type=Path, required=True, help="cf_cot 가 담긴 precompute 출력 JSON")
    p.add_argument("--output", type=Path, default=Path("results/judge_compare/ref_vs_cand.json"))
    p.add_argument("--ref-model", default="claude-opus-4-8", help="기준 judge 모델 (변별력 있는 강력 모델)")
    p.add_argument("--cand-model", default="solar-pro3", help="후보 judge 모델 (채택 검토 대상)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 호환 백엔드 base_url")
    p.add_argument("--key-env", default=DEFAULT_KEY_ENV, help="API 키 환경변수 이름")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-reps", type=int, default=1)
    p.add_argument("--judge-temp", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--id4-letter", default="A", help="id4 sanity 기대 supported_letter (원본 채점)")
    p.add_argument("--reuse-cand-from", default=None,
                   help="이전 리포트 JSON 경로 — 그 안의 cand(solar) 결과를 재사용해 cand 호출 생략(비용 절반)")
    p.add_argument("--abort-after-fails", type=int, default=8,
                   help="연속 N건 API 실패(토큰소진 등) 시 즉시 중단 (완료분은 체크포인트에 보존)")
    p.add_argument("--ref-only", action="store_true",
                   help="cand(후보) 채점 생략, ref 채점만 수행 (다른 서버에서 ref judge 점수만 뽑아올 때). "
                        "결과의 per_sample[].ds 에 ref judge 결과가 담긴다.")
    args = p.parse_args()

    samples = load_samples(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]

    # cf_cot 추출 — precompute metadata 에서
    work = []
    skipped = 0
    for s in samples:
        meta = (s.metadata or {}).get("counterfactual") or {}
        cfs = [c for c in meta.get("counterfactuals", []) if c.get("cf_cot")]
        if not cfs:
            skipped += 1
            continue
        work.append((s, cfs, meta.get("k", 1)))
    print(f"[compare] 입력 {len(samples)}건 중 cf_cot 보유 {len(work)}건 사용 (cf_cot 없음 {skipped}건 제외)")
    if not work:
        raise SystemExit("cf_cot 가 있는 샘플이 없습니다. precompute 출력(필터링 전 전체)을 입력하세요.")

    ref_client = _make_client(args.base_url, args.key_env)
    cand_client = _make_client(args.base_url, args.key_env)
    print(f"[compare] ref={args.ref_model}  cand={args.cand_model}  via {args.base_url}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 후보(cand) 결과 재사용 — 이전 리포트의 cand(solar) 채점을 그대로 사용 → cand 호출 생략
    reuse_cand = {}
    if args.reuse_cand_from and Path(args.reuse_cand_from).exists():
        prev = json.load(open(args.reuse_cand_from, encoding="utf-8"))
        for r in prev.get("per_sample", []):
            if isinstance(r.get("solar"), dict) and r["solar"].get("orig"):
                reuse_cand[str(r["id"])] = r["solar"]
        print(f"[compare] cand 재사용 {len(reuse_cand)}건 ← {args.reuse_cand_from}")

    # 체크포인트 — 완료 샘플을 JSONL 로 append. 재시작 시 해당 id 건너뜀.
    ckpt_path = args.output.with_suffix(".ckpt.jsonl")
    done_by_id = {}
    if ckpt_path.exists():
        for line in open(ckpt_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rr = json.loads(line)
                done_by_id[str(rr["id"])] = rr
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[compare] 체크포인트 {len(done_by_id)}건 재사용 ← {ckpt_path}")

    results: dict[int, dict] = {}
    todo = []
    for i, (s, _cfs, _k) in enumerate(work):
        if str(s.id) in done_by_id:
            results[i] = done_by_id[str(s.id)]
        else:
            todo.append(i)
    print(f"[compare] 신규 처리 {len(todo)}건 (체크포인트 {len(results)}건 건너뜀)")

    done = {"n": len(results)}
    consec_fail = {"n": 0}
    lock = threading.Lock()
    abort = threading.Event()
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    def _stub_judged(cfs):
        none = {"supported_letter": None, "matches_target": None, "score": None, "n_ok": 0}
        return {"orig": dict(none),
                "cfs": [{"target_idx": c["target_idx"], "target_letter": c["target_letter"],
                         "cf_cot": c["cf_cot"], **none} for c in cfs],
                "api_dead": False, "last_err": None}

    def _process(i: int):
        sample, cfs, k = work[i]
        ds = judge_sample(ref_client, args.ref_model, sample, cfs, reps=args.judge_reps, temp=args.judge_temp)
        sid = str(sample.id)
        if args.ref_only:
            sol = _stub_judged(cfs)               # cand 생략 — ref 점수만 수집
        elif sid in reuse_cand:
            sol = reuse_cand[sid]
        else:
            sol = judge_sample(cand_client, args.cand_model, sample, cfs, reps=args.judge_reps, temp=args.judge_temp)
        return i, {
            "id": sample.id,
            "ds": ds, "solar": sol,
            "ds_keep": decide_keep(sample, ds, k),
            "solar_keep": decide_keep(sample, sol, k),
        }

    def _log(i, r):
        with lock:
            done["n"] += 1
            ckpt_f.write(json.dumps(r, ensure_ascii=False) + "\n"); ckpt_f.flush()
            ref_dead = bool(r["ds"].get("api_dead"))
            cand_dead = bool(r["solar"].get("api_dead")) if isinstance(r["solar"], dict) else False
            consec_fail["n"] = consec_fail["n"] + 1 if (ref_dead or cand_dead) else 0
            tag = ""
            if ref_dead or cand_dead:
                err = r["ds"].get("last_err") or (r["solar"].get("last_err") if isinstance(r["solar"], dict) else None)
                tag = f"  ⚠️API실패{'[ref]' if ref_dead else ''}{'[cand]' if cand_dead else ''} {err}"
            print(f"[compare] {done['n']}/{len(work)} id={r['id']} "
                  f"ds_keep={r['ds_keep']} solar_keep={r['solar_keep']}{tag}")
            if consec_fail["n"] >= args.abort_after_fails and not abort.is_set():
                print(f"\n🛑 연속 {consec_fail['n']}건 API 실패 — 토큰 소진/장애 의심. 중단합니다. "
                      f"(완료 {done['n']}건은 {ckpt_path} 에 보존)")
                abort.set()

    if not todo:
        print("[compare] 신규 처리 없음 — 체크포인트만으로 리포트 생성.")
    elif args.workers <= 1:
        for i in todo:
            if abort.is_set():
                break
            idx, r = _process(i); results[idx] = r; _log(idx, r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process, i): i for i in todo}
            try:
                for fut in as_completed(futs):
                    idx, r = fut.result(); results[idx] = r; _log(idx, r)
                    if abort.is_set():
                        break
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    ckpt_f.close()

    ordered = [results[i] for i in range(len(work)) if i in results]
    if not ordered:
        raise SystemExit("처리된 샘플이 없습니다.")
    if abort.is_set() or len(ordered) < len(work):
        print(f"[compare] ⚠️ 부분 결과 {len(ordered)}/{len(work)}건으로 분석 "
              f"(체크포인트 {ckpt_path} — 토큰 충전 후 같은 명령 재실행 시 이어서 처리)")

    # ── 인스턴스 단위 수집 (원본/CF 태깅) ──────────────────────────────────────
    inst = []  # {type, dl, sl, dm, sm, dsc, ssc}
    for r in ordered:
        o_d, o_s = r["ds"]["orig"], r["solar"]["orig"]
        inst.append({"type": "orig", "dl": o_d["supported_letter"], "sl": o_s["supported_letter"],
                     "dm": o_d["matches_target"], "sm": o_s["matches_target"],
                     "dsc": o_d["score"], "ssc": o_s["score"]})
        for dc, sc in zip(r["ds"]["cfs"], r["solar"]["cfs"]):
            inst.append({"type": "cf", "dl": dc["supported_letter"], "sl": sc["supported_letter"],
                         "dm": dc["matches_target"], "sm": sc["matches_target"],
                         "dsc": dc["score"], "ssc": sc["score"]})

    def agree(items, ka, kb):
        pairs = [(i[ka], i[kb]) for i in items if i[ka] is not None and i[kb] is not None]
        if not pairs:
            return None, 0
        return sum(1 for x, y in pairs if x == y) / len(pairs), len(pairs)

    # ① supported_letter 일치율 (SWAP 판정) — 전체 + 진단용 orig/cf 분리
    sup_rate, sup_n = agree(inst, "dl", "sl")
    kappa = cohen_kappa([i["dl"] for i in inst], [i["sl"] for i in inst])
    sup_orig, sup_orig_n = agree([i for i in inst if i["type"] == "orig"], "dl", "sl")
    sup_cf, sup_cf_n = agree([i for i in inst if i["type"] == "cf"], "dl", "sl")

    # ② reject 결정 일치율 (SWAP 판정)
    keep_pairs = [(r["ds_keep"], r["solar_keep"]) for r in ordered]
    reject_rate = sum(1 for a, b in keep_pairs if a == b) / len(keep_pairs)
    ds_reject = {r["id"] for r in ordered if not r["ds_keep"]}
    sol_reject = {r["id"] for r in ordered if not r["solar_keep"]}
    union = len(ds_reject | sol_reject)
    reject_jaccard = (len(ds_reject & sol_reject) / union) if union else 1.0

    # id4 sanity (원본 채점 supported_letter)
    id4 = next((r for r in ordered if str(r["id"]) == "4"), None)
    id4_info = None
    if id4 is not None:
        id4_info = {"ds_supported": id4["ds"]["orig"]["supported_letter"],
                    "solar_supported": id4["solar"]["orig"]["supported_letter"],
                    "expected": args.id4_letter}

    # 진단용(참고) — 판정엔 미반영
    mt_rate, mt_n = agree(inst, "dm", "sm")
    mt_orig, _ = agree([i for i in inst if i["type"] == "orig"], "dm", "sm")
    mt_cf, _ = agree([i for i in inst if i["type"] == "cf"], "dm", "sm")
    spearman = None
    sp = [(i["dsc"], i["ssc"]) for i in inst if i["dsc"] is not None and i["ssc"] is not None]
    if len(sp) >= 3:
        from scipy.stats import spearmanr
        rho, _ = spearmanr([x for x, _ in sp], [y for _, y in sp])
        spearman = float(rho)

    # ── SWAP 판정 (①②+id4 만 사용) ────────────────────────────────────────────
    def ok(v, thr):
        return v is not None and v >= thr

    id4_pass = (bool(id4_info) and id4_info["solar_supported"] == args.id4_letter
                and id4_info["ds_supported"] == args.id4_letter)
    swap_checks = {
        "supported_letter": {"value": sup_rate, "n": sup_n, "kappa": kappa,
                             "pass": ok(sup_rate, 0.80), "thr": ">=80%"},
        "reject_decision": {"value": reject_rate, "reject_jaccard": reject_jaccard,
                            "pass": ok(reject_rate, 0.85), "thr": ">=85%"},
        "id4_sanity": {"value": id4_info, "pass": id4_pass,
                       "thr": f'solar & ds supported_letter == "{args.id4_letter}"'},
    }
    verdict = all(c["pass"] for c in swap_checks.values())

    # borderline 감지 — 판정 지표가 임계값 ±0.05 안이면 진단(③) 확인 권고
    def borderline(v, thr):
        return v is not None and abs(v - thr) <= 0.05
    is_borderline = borderline(sup_rate, 0.80) or borderline(reject_rate, 0.85)

    diagnostics = {
        "supported_letter_split": {"orig": sup_orig, "orig_n": sup_orig_n,
                                   "cf": sup_cf, "cf_n": sup_cf_n},
        "matches_target": {"overall": mt_rate, "orig": mt_orig, "cf": mt_cf, "n": mt_n},
        "score_spearman": spearman,
        "reject_jaccard": reject_jaccard,
        "ds_reject_ids": sorted(ds_reject, key=str), "solar_reject_ids": sorted(sol_reject, key=str),
    }

    report = {
        "ref_model": args.ref_model, "cand_model": args.cand_model,
        "n_samples": len(work), "n_instances": len(inst), "judge_reps": args.judge_reps,
        "swap_checks": swap_checks,
        "VERDICT": f"ADOPT {args.cand_model}" if verdict else f"FAIL — do not adopt {args.cand_model}",
        "borderline": is_borderline,
        "diagnostics": diagnostics,
        "per_sample": ordered,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 콘솔 요약 ──────────────────────────────────────────────────────────────
    def fmt(x): return "N/A" if x is None else f"{x:.3f}"
    def mark(b): return "통과" if b else "실패"
    c = swap_checks
    print("\n" + "=" * 70)
    print(f"JUDGE SWAP 판정 — ref={args.ref_model}  vs  cand={args.cand_model}")
    print("=" * 70)
    print(f"  샘플 {len(work)}건 / 채점 인스턴스 {len(inst)}개 (원본+CF), reps={args.judge_reps}")
    print("  [SWAP 결정용]")
    print(f"   ① supported_letter 일치율 : {fmt(c['supported_letter']['value'])}  "
          f"(kappa={fmt(kappa)})  [{mark(c['supported_letter']['pass'])}]  기준 ≥80%")
    print(f"   ② reject 결정 일치율       : {fmt(c['reject_decision']['value'])}  "
          f"(Jaccard={fmt(reject_jaccard)})  [{mark(c['reject_decision']['pass'])}]  기준 ≥85%")
    print(f"   + id4 sanity              : {id4_info}  [{mark(id4_pass)}]")
    print("  [진단용 — 참고]")
    print(f"   ③ supported_letter 분리   : orig={fmt(sup_orig)}({sup_orig_n})  cf={fmt(sup_cf)}({sup_cf_n})")
    print(f"      matches_target         : overall={fmt(mt_rate)}  orig={fmt(mt_orig)}  cf={fmt(mt_cf)}")
    print(f"      score Spearman rho     : {fmt(spearman)}")
    print("-" * 70)
    print(f"  최종 판정: {report['VERDICT']}" + ("   ⚠️ borderline — 진단 ③ 확인 권장" if is_borderline else ""))
    print("=" * 70)
    print(f"리포트 저장 → {args.output}")


if __name__ == "__main__":
    main()
