import argparse
import json
import sys
import numpy as np
from pathlib import Path
from filters.kppl.run import KPPLFilter
from utils.schema import CoTSample
from utils.data_loader import load_samples

# Windows cp949 콘솔에서 유니코드 출력
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp949", "cp1252", "mbcs"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser(description="PPL 분포 확인용 스크립트")
parser.add_argument("--input", type=Path, default=Path("data/raw/results_deepseek-r1_cot_forced_300.jsonl"))
parser.add_argument("--max_samples", type=int, default=None, help="진단용: 처음 N개만 측정")
args = parser.parse_args()

samples = load_samples(args.input)
if args.max_samples:
    samples = samples[:args.max_samples]

kppl_filter = KPPLFilter(low_threshold=0.0, high_threshold=float("inf"))

ppls = []
skipped = 0
errors: list[str] = []
for i, sample in enumerate(samples):
    _, ppl, reason = kppl_filter.filter_sample(sample)
    if ppl is not None:
        ppls.append(ppl)
    else:
        skipped += 1
        if len(errors) < 5:  # 첫 5개 에러만 출력
            errors.append(f"  sample[{i}] id={sample.id}: {reason}")

if skipped:
    print(f"[경고] PPL 계산 실패 샘플 {skipped}개 제외됨 - cot 비어있거나 인코딩 오류 확인 필요")
    for e in errors:
        print(e)

ppls = np.array(ppls)
print(f"유효 샘플 수: {len(ppls)} / {len(samples)}")

if len(ppls) == 0:
    print("[오류] 유효한 PPL 샘플이 없습니다. 입력 데이터를 확인하세요.")
else:
    n = len(ppls)
    print(f"min:    {ppls.min():.2f}")
    print(f"p10:    {np.percentile(ppls, 10):.2f}")
    print(f"p25:    {np.percentile(ppls, 25):.2f}")
    print(f"median: {np.median(ppls):.2f}")
    print(f"p75:    {np.percentile(ppls, 75):.2f}")
    print(f"p90:    {np.percentile(ppls, 90):.2f}")
    print(f"max:    {ppls.max():.2f}")

    print(f"\n{'─'*52}")
    print(f"{'Threshold (low / high)':<24} {'탈락':>5} {'통과':>5} {'탈락률':>7}")
    print(f"{'─'*52}")
    candidates = [
        (3.0, 7.5), (3.5, 7.0), (3.5, 7.5),
        (3.6, 6.5), (3.6, 7.0), (4.0, 6.5), (4.0, 7.0),
    ]
    for low, high in candidates:
        rejected = int(((ppls < low) | (ppls > high)).sum())
        passed = n - rejected
        print(f"  low={low:.1f}, high={high:.1f}          {rejected:>5} {passed:>5} {rejected/n:>7.1%}")
    print(f"{'─'*52}")