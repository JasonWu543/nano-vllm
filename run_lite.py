import torch
from transformers import AutoTokenizer
from mla_absorb import YoutuForCausalLM, YoutuConfig, load_model_first_layer
import os
import random
import numpy as np
from transformers import PreTrainedTokenizerFast
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
# 1. 实例化 Lite 配置
config = YoutuConfig(
    num_hidden_layers=1, # 强制只初始化一层以节省显存
    q_lora_rank=None,    # Lite 不压缩 Q
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    hidden_size=2048,
    intermediate_size=10944
)

# 2. 创建模型
model = YoutuForCausalLM(config)

# 3. 加载权重 (请修改为你的实际路径)
ckpt_path = os.path.abspath("/root/autodl-tmp/models")
load_model_first_layer(model, ckpt_path)

# 4. Tokenizer 处理文本


tokenizer_file = os.path.join(ckpt_path, "tokenizer.json")

path = "/root/autodl-tmp/models/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(path)
prompt = "Introduce yourself"
inputs = tokenizer(prompt, return_tensors="pt")
# 5. 推理 (确保已删掉 mla_absorb.py 里的 exit(-1))
model.eval()
with torch.no_grad():
    outputs = model(input_ids=inputs["input_ids"])
    print(f"\n输入: {prompt}")
    print(f"第一层输出 Hidden States Shape: {outputs.logits.shape}")