"""
K-PPL Band-pass Filter  (③ Ablation — 언어 품질)

EXAONE-3.5-2.4B로 CoT의 Perplexity를 계산하여 Band-pass 필터링.
PPL이 너무 낮으면(단순 반복), 너무 높으면(번역체·비문) 제거.

담당: B
참고: Perplexed by Perplexity (arXiv:2405.20541)
"""

from __future__ import annotations

import math
import argparse
from pathlib import Path

import yaml

from filters.base import BaseFilter
from utils.schema import CoTSample


_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"


def _load_kppl_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["ablation_filters"]["kppl"]


class KPPLFilter(BaseFilter):
    name = "kppl"

    def __init__(
        self,
        model_name: str | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        device: str | None = None,
        max_length: int | None = None,
        stride: int | None = None,
    ):
        cfg = _load_kppl_config()
        model_name     = model_name     if model_name     is not None else cfg.get("model_name",     "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
        low_threshold  = low_threshold  if low_threshold  is not None else cfg.get("low_threshold",  5.0)
        high_threshold = high_threshold if high_threshold is not None else cfg.get("high_threshold", 50.0)
        device         = device         if device         is not None else cfg.get("device",         "cuda")
        max_length     = max_length     if max_length     is not None else cfg.get("max_length",     2048)
        stride         = stride         if stride         is not None else cfg.get("stride",         512)
        if stride > max_length:
            raise ValueError(
                f"stride({stride})은 max_length({max_length}) 이하여야 합니다. "
                "stride > max_length이면 윈도우 간 갭이 생겨 토큰이 누락됩니다."
            )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            self.device = device
            print("[KPPLFilter] CUDA not available, falling back to CPU")

        # revision e949c91dec92: transformers 4.46-4.47 호환 버전
        # 이후 revision은 RopeParameters(4.48+) 요구로 4.46-4.47과 충돌
        revision = "e949c91dec92"
        print(f"[KPPLFilter] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, revision=revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        print(f"[KPPLFilter] Model loaded on {device}")

    def _compute_ppl(self, text: str) -> float:
        """
        단일 텍스트의 PPL을 Sliding Window로 계산한다.

        HuggingFace cross-entropy loss는 자연로그(nats) 기반 NLL을 반환하므로
        PPL = exp(NLL_e)로 변환한다.
        로그 밑과 지수 밑은 반드시 함께 일치시켜야 동일한 PPL 값을 얻는다:
          - NLL을 자연로그(e) 기반으로 계산 → exp() 사용  ← 여기서 채택
          - NLL을 이진로그(2) 기반으로 계산 → 2**() 사용
        밑수만 바꾸고 NLL 단위를 바꾸지 않으면 전혀 다른 값이 산출된다.
        """
        import torch

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids  # shape: (1, seq_len)
        seq_len = input_ids.size(1)

        if seq_len <= self.max_length:
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            return float(torch.exp(outputs.loss).item())

        # Sliding Window: 긴 텍스트 처리
        # prev_end_loc 을 추적해 새로 추가된 타깃 토큰만 집계.
        #
        # HF causal LM 의 loss 는 내부적으로 shift 후 (W-1)개 위치에서 계산되며,
        # labels[:, :context_len] = -100 로 마스킹된 뒤 비마스킹 타깃 수는
        #     valid_count = W - max(context_len, 1)
        # 이다. 첫 윈도우(context_len=0)에선 (W-1) 개, 후속 윈도우에선 stride 개.
        # 기존 코드는 valid_count = end - prev_end_loc 로 매 윈도우 +1 씩 over-count
        # 하여 PPL 을 미세하게 낮게 추정했다. 수정 후 loss 가중 평균과 분자/분모
        # 단위가 일치한다.
        total_nll = 0.0
        total_tokens = 0
        prev_end_loc = 0

        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_length, seq_len)
            window_size = end - begin
            window_ids = input_ids[:, begin:end].to(self.device)

            context_len = max(prev_end_loc - begin, 0)
            labels = window_ids.clone()
            labels[:, :context_len] = -100

            # shift 후 실제 비마스킹 타깃 개수
            valid_target_count = window_size - max(context_len, 1)
            if valid_target_count <= 0:
                prev_end_loc = end
                if end == seq_len:
                    break
                continue

            with torch.no_grad():
                outputs = self.model(window_ids, labels=labels)

            total_nll += outputs.loss.item() * valid_target_count
            total_tokens += valid_target_count

            prev_end_loc = end
            if end == seq_len:
                break

        avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
        return float(math.exp(avg_nll))

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        text = sample.cot
        if not text or not text.strip():
            return False, None, "cot가 비어 있음"

        try:
            ppl = self._compute_ppl(text)
        except Exception as e:
            return False, None, f"PPL 계산 오류: {e}"

        if ppl < self.low_threshold:
            return False, ppl, f"PPL too low ({ppl:.2f} < {self.low_threshold}): 단순 반복 의심"
        if ppl > self.high_threshold:
            return False, ppl, f"PPL too high ({ppl:.2f} > {self.high_threshold}): 번역체·비문 의심"

        return True, ppl, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    KPPLFilter().run_from_file(args.input, args.output)
