import torch
import torch.nn.functional as F
from einops import einsum, repeat

import tilelang as tl
import tilelang.language as T
from typing import Optional


@torch.no_grad()
def full_indexer_bwd_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    indexq: torch.Tensor,
    weights: torch.Tensor,
    indexk: torch.Tensor,
    offsets: torch.Tensor,
    chunk_size: int = 2048,
    eps: float = 1e-9,
):
    device = q.device
    softmax_scale = q.shape[-1] ** -0.5
    S, H, _ = q.shape #S总token数 H head数
    d = indexq.shape[-1]    #indexer的维度
    #创建梯度返回值
    dindexq = torch.zeros_like(indexq)
    dweights = torch.zeros_like(weights)
    dindexk = torch.zeros_like(indexk)

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    B = offsets.numel() - 1 #batch size batch里有多少条序列

    for bi in range(B): 
        start = int(offsets[bi].item())
        end = int(offsets[bi + 1].item())   #取出这个样本在packed张量中的起止位置
        seq_len = end - start   #计算长度
        if seq_len <= 0:
            continue

        q_batch = q[start:end]
        k_batch = k[start:end]
        indexq_batch = indexq[start:end]
        weights_batch = weights[start:end]
        indexk_batch = indexk[start:end]    #只取当前样本的token段

        for chunk_start in range(0, seq_len, chunk_size):   #按chunk切分    
            chunk_end = min(chunk_start + chunk_size, seq_len)  #这一段chunk的终点

            q_chunk = q_batch[chunk_start:chunk_end]    #取出对应的q
            k_full = k_batch[:chunk_end]                #casual attention，所以只用看到chunk_end
            IQ = indexq_batch[chunk_start:chunk_end]    #取出对应的Indexq
            W = weights_batch[chunk_start:chunk_end]    #取出对应的权重
            IK = indexk_batch[:chunk_end]               #取出Indexk

            s1 = chunk_end - chunk_start    #本chunk里的query数量
            s2 = chunk_end                  #本chunk里key的数量

            qp = torch.arange(chunk_start, chunk_end, device=device)[:, None]   #query position 在样本中的绝对位置
            kp = torch.arange(chunk_end, device=device)[None, :]    #key的绝对位置
            causal_2d = (qp >= kp)  #构造casual mask

            attn_logits = einsum(q_chunk, k_full, 'q h d, k d -> q h k') * softmax_scale    #标准注意力里的Q*K
            attn_logits = attn_logits.masked_fill(~causal_2d.unsqueeze(1), float('-inf'))   #非法key位置应用mask
            attn_prob_h = torch.softmax(attn_logits, dim=-1)    #softmax得到每个head的注意力分布

            p = attn_prob_h.sum(dim=1)  #把所有head的注意力加起来
            p = p / (p.sum(dim=-1, keepdim=True) + eps) #再归一化

            #统一进行截断？
            logp_clip = (p.clamp_min(eps).log()).clamp(-100.0, 0.0) 
            p_used = logp_clip.exp().to(torch.float32)

            T = einsum(IQ, IK, 'i h k, j k -> i h j') * softmax_scale  #indexer打分
            relu_mask = (T > 0) #RELU 只保留正的
            R = torch.relu(T)
        

            Sij = (R * W.unsqueeze(-1)).sum(dim=1).to(torch.float32)    #对每个头加权求和
            U = Sij.masked_fill(~causal_2d, float('-inf'))  #因果mask
            logq = torch.log_softmax(U, dim=-1) #softmax的另一种计算方式
            qhat = logq.exp()
            #截断
            in_range = (logq > -100.0).to(torch.float32)
            logq_clip = torch.maximum(
                logq,
                torch.tensor(-100.0, device=device, dtype=logq.dtype),
            )

            loss = (p_used * (logp_clip.to(torch.float32) - logq_clip.to(torch.float32))).sum() #计算KL散度
            total_loss += loss  #累加loss

            g = (-p_used * in_range).masked_fill(~causal_2d, 0.0)   #loss 对logq的梯度
             #Loss对U的梯度
            g_sum = g.sum(dim=-1, keepdim=True)
            dU = g - qhat.to(torch.float32) * g_sum
            dSij = dU   #梯度一样

            dW = (dSij.unsqueeze(1) * R.to(torch.float32)).sum(dim=-1)  #对w的梯度
            dR = dSij.unsqueeze(1) * W.to(torch.float32).unsqueeze(-1)  #对R的梯度
            dT = dR * relu_mask.to(torch.float32)   #对T的梯度

            dIQ = softmax_scale * einsum(dT, IK.to(torch.float32), 'i h j, j k -> i h k')   #反传dIQ
            dIK = softmax_scale * einsum(dT, IQ.to(torch.float32), 'i h j, i h k -> j k')   #反传dIK

            dindexq[start + chunk_start:start + chunk_end] += dIQ.to(dindexq.dtype) #把本chunk的累加回全局梯度
            dweights[start + chunk_start:start + chunk_end] += dW.to(dweights.dtype)
            dindexk[start:start + chunk_end] += dIK.to(dindexk.dtype)
    
    return dindexq, dweights, dindexk   #返回
    
