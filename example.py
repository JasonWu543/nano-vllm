import os
# 禁用 torch.compile，直接返回原函数/原模块
def _fake_torch_compile(model=None, *args, **kwargs):
    return model
import torch
torch.compile = _fake_torch_compile
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

import torch.distributed as dist

# 单卡兜底 patch
dist.get_rank = lambda *args, **kwargs: 0
dist.get_world_size = lambda *args, **kwargs: 1
dist.is_initialized = lambda: False
dist.barrier = lambda *args, **kwargs: None
dist.all_reduce = lambda tensor, *args, **kwargs: tensor

def _fake_all_gather(output_list, tensor, *args, **kwargs):
    if output_list and len(output_list) > 0:
        output_list[0].copy_(tensor)
dist.all_gather = _fake_all_gather

dist.broadcast = lambda tensor, *args, **kwargs: tensor
dist.init_process_group = lambda *args, **kwargs: None
dist.destroy_process_group = lambda *args, **kwargs: None
from nanovllm import LLM, SamplingParams
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 CuDNN 使用确定性算法（可能会略微降低性能，但数值会完全一致）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 环境变量设置（可选，防止一些特定的算子出现随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 随机种子已固定为: {seed}")

set_seed(42)
def main():
    path = "/root/autodl-tmp/models/DeepSeek-V2-Lite"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
    path, 
    tensor_parallel_size=1,
    max_model_len=4096,            
    max_num_batched_tokens=4096,    
    gpu_memory_utilization=0.5,    
)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=32)
    prompts = [
        "Introduce yourself",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        print(f"Token IDs: {output['token_ids']}")
        print(f"Token count: {len(output['token_ids'])}")


if __name__ == "__main__":
    main()
