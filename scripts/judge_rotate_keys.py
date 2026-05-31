"""
precompute judge 를 여러 API 키로 자동 순회 채점하는 드라이버.

무료/소액 키는 금방 쿼터(insufficient_quota)가 나는데, precompute 는 체크포인트에
성공분만 저장하고 실패분은 재시도 대상으로 남긴다. 이 스크립트는 키 목록을 돌며
같은 precompute 명령을 반복 실행해서, 한 키가 죽으면 다음 키로 이어서 채점한다.
전부(=입력 샘플 수) 채점되면 멈춘다.

키는 코드/깃에 박지 말고 파일에 한 줄에 하나씩 둔다(`#` 주석/빈 줄 무시).
OPENAI_BASE_URL / OPENAI_JUDGE_MODEL 은 부모 환경에서 상속되므로, 실행 전에 한 번만
설정하면 모든 키에 공통 적용된다(키만 회전).

사용:
    # (먼저 공통 엔드포인트/모델 설정 — 게이트웨이 예시)
    $env:OPENAI_BASE_URL="https://factchat-cloud.mindlogic.ai/v1/gateway"
    $env:OPENAI_JUDGE_MODEL="gpt-5"

    python scripts/judge_rotate_keys.py `
        --input  data/counterfactual/train_cf.jsonl `
        --output data/counterfactual/cf_judged.json `
        --keys-file keys.txt `
        --workers 4            # rep1 (기본). rep3 면 --judge-reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _count_input(path: Path) -> int:
    if path.suffix == ".jsonl":
        return sum(1 for line in open(path, encoding="utf-8") if line.strip())
    data = json.load(open(path, encoding="utf-8"))
    return len(data) if isinstance(data, list) else 1


def _count_done(ckpt_path: Path) -> int:
    """체크포인트에 저장된 (성공) 샘플의 고유 인덱스 수."""
    if not ckpt_path.exists():
        return 0
    seen = set()
    for line in open(ckpt_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(int(json.loads(line)["i"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return len(seen)


def _load_keys(path: Path) -> list[str]:
    keys = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    return keys


def _run_key(base_cmd: list[str], env: dict, dead_threshold: int) -> str:
    """precompute 를 한 키로 실행하며 출력을 실시간 스트리밍.

    쿼터/인증 실패로 ERROR 줄이 연속 dead_threshold 개 나오면 = 이 키는 죽은 것 →
    남은 수천 건을 계속 시도하며 시간 낭비하지 않도록 subprocess 를 즉시 종료한다.
    반환: "dead"(연속실패로 강제종료) | "done"(자연종료).
    """
    # 자식이 한글 로그를 파이프로 출력할 때 cp949 로 죽지 않게 강제 UTF-8.
    env = {**env, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        base_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    consec_err = 0
    killed = False
    try:
        for line in proc.stdout:
            print(line, end="")
            if "ERROR=" in line:
                consec_err += 1
                if consec_err >= dead_threshold:
                    print(f"[rotate] ↳ 연속 실패 {consec_err}건 → 이 키 소진으로 판단, 종료하고 다음 키로.")
                    proc.terminate()
                    killed = True
                    break
            elif "cf_scores=" in line:   # 성공 진행 줄 → 실패 카운터 리셋
                consec_err = 0
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdout:
            proc.stdout.close()
    return "dead" if killed else "done"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keys-file", type=Path, required=True,
                   help="API 키 목록 파일(한 줄에 하나, # 주석 무시). 깃에 커밋 금지.")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--judge-reps", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--stop-after-dead", type=int, default=0,
                   help="연속으로 진행 0건인 키가 이 수만큼 나오면 중단(0=비활성). 죽은 키만 남았을 때 무한낭비 방지.")
    p.add_argument("--dead-threshold", type=int, default=12,
                   help="한 키 실행 중 ERROR 줄이 연속 이 수만큼 나오면 키 소진으로 보고 즉시 종료→다음 키. (workers 보다 넉넉히)")
    args = p.parse_args()

    total = _count_input(args.input)
    ckpt = args.output.with_suffix(".ckpt.jsonl")
    keys = _load_keys(args.keys_file)
    if not keys:
        sys.exit(f"[rotate] 키가 없습니다: {args.keys_file}")

    base_cmd = [
        sys.executable, "-m", "filters.counterfactual.precompute",
        "--input", str(args.input), "--output", str(args.output),
        "--k", str(args.k), "--judge-reps", str(args.judge_reps),
        "--workers", str(args.workers), "--sleep", str(args.sleep),
    ]

    done0 = _count_done(ckpt)
    print(f"[rotate] 입력 {total}건 | 시작 시 완료 {done0}건 | 키 {len(keys)}개 | 남은 ~{total - done0}건")
    if done0 >= total:
        print("[rotate] 이미 전부 채점됨. 종료."); return

    dead_streak = 0
    for idx, key in enumerate(keys, 1):
        before = _count_done(ckpt)
        if before >= total:
            print(f"[rotate] 전부 완료({before}/{total}). 남은 키 {len(keys) - idx + 1}개 미사용.")
            break

        env = dict(os.environ)
        env["OPENAI_API_KEY"] = key
        masked = key[:6] + "…" + key[-4:] if len(key) > 12 else "****"
        print(f"\n[rotate] === 키 {idx}/{len(keys)} ({masked}) 시작 | 현재 {before}/{total} ===")
        # precompute 는 키가 죽어도 남은 전건을 계속 시도하므로, 연속 실패를 감지해 조기 종료한다.
        _run_key(base_cmd, env, args.dead_threshold)

        after = _count_done(ckpt)
        gained = after - before
        print(f"[rotate] 키 {idx} 종료: +{gained}건 (누적 {after}/{total})")

        if after >= total:
            print(f"\n[rotate] ✅ 전부 채점 완료 {after}/{total}")
            return
        dead_streak = dead_streak + 1 if gained == 0 else 0
        if args.stop_after_dead and dead_streak >= args.stop_after_dead:
            print(f"[rotate] ⚠ 연속 {dead_streak}개 키가 진행 0건 → 중단. 새 키를 채워 다시 실행하세요.")
            return

    final = _count_done(ckpt)
    if final < total:
        print(f"\n[rotate] ⚠ 키 소진. {final}/{total} 완료, {total - final}건 남음. "
              f"새 키를 {args.keys_file} 에 추가하고 같은 명령 재실행하면 이어집니다.")


if __name__ == "__main__":
    main()
