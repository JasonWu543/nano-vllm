import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

#TODO：目前存了K和V，后面把存V的相关代码删了
#MODEL runner也跟着改了一部分
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,        
    slot_mapping_ptr,
    D,              #real D
    BLOCK_D: tl.constexpr,#padded D
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    offsets_d=tl.arange(0,BLOCK_D)
    mask=offsets_d<D
    key_offsets = idx * key_stride + offsets_d
    value_offsets = idx * value_stride + offsets_d
    key = tl.load(key_ptr + key_offsets,mask=mask)
    value = tl.load(value_ptr + value_offsets,mask=mask)
    cache_offsets = slot * D + offsets_d
    tl.store(k_cache_ptr + cache_offsets, key ,mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value ,mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    if key.dim()==3:
        key=key[:,0,:]
        value=value[:,0,:]  #only head0
    N,D=key.shape
    #padding
    BLOCK_D=triton.next_power_of_2(D)

    if key.stride(-1)!=1:
        key=key.contiguous()
    if value.stride(-1)!=1:
        value=value.contiguous()
    
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D ,BLOCK_D=BLOCK_D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache #TODO 把这的prefill换成flash attn
            o_prefill = eager_attention_forward_varlen(
                q=q,
                k=k,
                v=v,
                cu_seqlens=context.cu_seqlens_q,
                scale=self.scale
            )
            return o_prefill[..., :512]            

            
         
        else:   
           
            cache_seqlens=context.context_lens.to(torch.int32)
            block_tables=context.block_tables.to(torch.int32)
            batch_size=q.shape[0]
            k_cache_4d=self.k_cache.unsqueeze(2).contiguous()
            q_4d=q.unsqueeze(1).contiguous()
            assert q_4d.dim()==4
            assert q.shape[1]==self.num_heads
            assert q.dtype==torch.bfloat16

            assert self.k_cache_4d.dim()==4
            assert self.k_cache.size(1)==64
            assert self.k_cache.dtype==torch.bfloat16

            assert block_tables.dtype==torch.int32
            assert cache_seqlens.dtype==torch.int32

            num_q_tokens_per_head_k=self.num_heads

            tile_scheduler_metadata,num_splits =get_mla_metadata(
                cache_seqlens,
                num_q_tokens_per_head_k,
                1
            )
            dv=512
            o,_=flash_mla_with_kvcache(
                q_4d,
                k_cache_4d,
                block_tables,
                cache_seqlens,
                dv,
                tile_scheduler_metadata,
                num_splits,
                causal=True
            )
        return o.squeeze(1)


def eager_attention_forward_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    手动实现的变长序列 (Varlen) Eager Attention (纯矩阵乘法)
    适用于绕过 FlashAttention 维度限制的 Prefill 阶段。
    
    参数:
        q, k, v: 形状均为 [total_tokens, num_heads, head_dim]
        cu_seqlens: 形状为 [batch_size + 1]，记录每个句子的起止索引
        scale: softmax 缩放因子
    返回:
        output: 形状为 [total_tokens, num_heads, head_dim]
    """
    # 准备一个和 q 形状一样的空张量，用来存放最终结果
    output = torch.empty_like(q)

    # 遍历当前 batch 中的每一个句子
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        seq_len = end - start
        
        if seq_len == 0:
            continue

        # 1. 截取当前句子的 Token
        # 形状转换: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q_i = q[start:end].transpose(0, 1)
        k_i = k[start:end].transpose(0, 1)
        v_i = v[start:end].transpose(0, 1)

        # 2. 计算 Attention Score (Q * K^T)
        # k_i.transpose(-1, -2) 会把最后两维倒过来，方便矩阵乘法
        # 结果形状: [num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale

        # 3. 构造并施加因果掩码 (Causal Mask)
        # 生成一个上三角为 -inf 的矩阵，防止当前 token 看到未来的 token
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device, dtype=attn_weights.dtype),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        # 4. Softmax 归一化 (强制使用 float32 保证数值稳定性，算完转回原数据类型)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # 5. 加权求和 (乘以 Value)
        # 结果形状: [num_heads, seq_len, head_dim]
        out_i = torch.matmul(attn_weights, v_i)

        # 6. 转置回 [seq_len, num_heads, head_dim] 并原路塞回总张量
        output[start:end] = out_i.transpose(0, 1)

    return output