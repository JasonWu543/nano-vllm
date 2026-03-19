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
def main():
    path = "/root/autodl-tmp/models/DeepSeek-V2-Lite"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
      "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
