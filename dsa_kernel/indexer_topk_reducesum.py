import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang as tl
import tilelang.language as T
from typing import Optional
from .index import prepare_token_indices


BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

def indexer_topk_reducesum_interface(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    offsets: torch.Tensor,
    dtype: str = BF16,
    chunk_size: int = 256,
):
    total_seq_len = q.shape[0]  #packed后的总token数
    device = q.device
    softmax_scale = q.shape[-1] ** -0.5
    
    all_topk_indices = torch.full((total_seq_len, topk), -1, dtype=torch.int32, device=device)  #创建两个输出张量
    all_topk_score = torch.full((total_seq_len, topk), float('-inf'), dtype=torch.float32, device=device)   
    
    for batch_idx in range(offsets.shape[0] - 1):
        start_idx = offsets[batch_idx].item()   #当前样本在packed张量中的范围
        end_idx = offsets[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        #把当前样本对应的q，weights，k切出来
        q_batch = q[start_idx:end_idx]
        weights_batch = weights[start_idx:end_idx]
        k_batch = k[start_idx:end_idx]
        #按chunk分层
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            
            q_chunk = q_batch[chunk_start:chunk_end]    #当前chunk的query
            weights_chunk = weights_batch[chunk_start:chunk_end]    #权重
            
            k_visible = k_batch[:chunk_end] #可见的key范围
            
            logits = einsum(q_chunk, k_visible, 's1 h d, s2 d -> s1 h s2')  #计算打分矩阵
            logits = F.relu(logits) #relu
            
            logits = (logits * weights_chunk.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale    #加权求和
            #构造causal mask
            row_indices = torch.arange(chunk_start, chunk_end, device=device)[:, None]
            col_indices = torch.arange(chunk_end, device=device)[None, :]
            mask = row_indices >= col_indices
            
            logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=device))
            
            if chunk_end < topk:    #若可见key数小于topk，补齐维度
                pad_size = topk - chunk_end
                logits = F.pad(logits, (0, pad_size), value=float('-inf'))
            
            topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)  #选择topk，并计算score
            topk_scores = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
            
            if chunk_end < topk:    #处理pad出来的无效项
                valid_mask = topk_indices < chunk_end
                topk_indices = torch.where(valid_mask, topk_indices, torch.tensor(-1, dtype=torch.int32, device=device))
                topk_scores = torch.where(valid_mask, topk_scores, torch.tensor(float(0.0), device=device))
            #写回全局
            global_start = start_idx + chunk_start
            global_end = start_idx + chunk_end
            all_topk_indices[global_start:global_end] = topk_indices
            all_topk_score[global_start:global_end] = topk_scores
    
    return all_topk_indices, all_topk_score