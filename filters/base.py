from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from utils.schema import CoTSample
from utils.data_loader import load_samples, save_samples


class BaseFilter(ABC):
    """
    모든 필터가 상속해야 하는 베이스 클래스.

    구현 방법:
        class CorrectnessFilter(BaseFilter):
            name = "correctness"

            def filter_sample(self, sample: CoTSample) -> tuple[bool, float | None, str | None]:
                passed = sample.is_correct
                return passed, None, None
    """

    name: str = "base"

    @abstractmethod
    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        """
        Returns:
            passed: 이 샘플이 필터를 통과했는지
            score:  필터가 계산한 점수 (없으면 None)
            reason: 제거 이유 (없으면 None)
        """
        ...

    def run(
        self,
        samples: list[CoTSample],
        verbose: bool = True,
    ) -> list[CoTSample]:
        passed_samples = []
        n_rejected = 0

        for sample in samples:
            passed, score, reason = self.filter_sample(sample)
            sample.add_filter_result(
                filter_name=self.name,
                passed=passed,
                score=score,
                reason=reason,
            )
            if passed:
                passed_samples.append(sample)
            else:
                n_rejected += 1

        rejection_rate = n_rejected / len(samples) if samples else 0
        if verbose:
            print(
                f"[{self.name}] "
                f"{len(samples)} → {len(passed_samples)} "
                f"(rejected {n_rejected}, {rejection_rate:.1%})"
            )
        return passed_samples

    def run_from_file(
        self,
        input_path: Path,
        output_path: Path,
        verbose: bool = True,
    ) -> list[CoTSample]:
        samples = load_samples(input_path)
        result = self.run(samples, verbose=verbose)
        save_samples(result, output_path)
        return result
