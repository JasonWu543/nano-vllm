import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    # 1. 彻底去掉这里的 @torch.compile (采样太轻，没必要编译，还容易出 RNG Bug)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 2. 确保 logits 是 [Batch, Vocab]
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        # 3. 临时改用 Greedy 采样绕过报错，验证复读问题
        # 如果这里改用 argmax 还是复读，那说明根本不是采样的问题，而是隐藏状态数值崩了
        return logits.argmax(dim=-1)