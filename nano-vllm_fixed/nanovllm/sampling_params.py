from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    greedy: bool = False

    def __post_init__(self):
        if self.greedy:
            self.temperature = 0.0
        else:
            assert self.temperature > 1e-10, "greedy sampling is not permitted"
