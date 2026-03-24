import torch
from torch import nn
import triton
import triton.language as tl
import torch.nn.functional as F
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
    k_stride=key.stride(0)
    v_stride=value.stride(0)
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
    store_kvcache_kernel[(N,)](key, k_stride, value, v_stride, k_cache, v_cache, slot_mapping, D ,BLOCK_D=BLOCK_D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_lora_rank,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_lora_rank = kv_lora_rank
        self._w_key: torch.Tensor | None = None
        self._w_vo: torch.Tensor | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
         
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache 
            #解压k
            kv_c=k[:,0,:self.kv_lora_rank]  #[N,512]
            k_rope=k[:,0,self.kv_lora_rank:]  #[N,64]
            kv_c_exp = kv_c.unsqueeze(0).expand(self.num_heads, -1, -1) #[H,N,512]
            w_key_T = self._w_key.transpose(1, 2)   #[H,512,64]
            k_nope = torch.bmm(kv_c_exp, w_key_T).transpose(0, 1)   #[N,H,128]
            k_rope =  k_rope.unsqueeze(1).expand(-1, self.num_heads, -1)    #[N,H,64]
            k_full=torch.cat([k_nope,k_rope],dim=-1).contiguous()    #[N,H,192]
            #解压v
            w_vo_T=self._w_vo.transpose(1, 2)   #[H,512,128]
            v_raw=torch.bmm(kv_c_exp, w_vo_T).transpose(0, 1).contiguous()  #[N,H,128]
            v_full = F.pad(v_raw, [0, 64])
            
            o = flash_attn_varlen_func(     q=q, 
                                            k=k_full, 
                                            v=v_full,
                                            max_seqlen_q=context.max_seqlen_q, 
                                            cu_seqlens_q=context.cu_seqlens_q,
                                            max_seqlen_k=context.max_seqlen_k,
                                            cu_seqlens_k=context.cu_seqlens_k,
                                            softmax_scale=self.scale,
                                            causal=True,
                                            block_table=context.block_tables)
            o = o[:, :, :128]
          
            return o          

            
         
        else:   
           
            cache_seqlens=context.context_lens.to(torch.int32)
            block_tables=context.block_tables.to(torch.int32)
            batch_size=q.shape[0]
            k_cache_4d=self.k_cache.unsqueeze(2).contiguous()
            q_4d=q.unsqueeze(1).contiguous()
            assert q_4d.dim()==4
            assert q.shape[1]==self.num_heads
            assert q.dtype==torch.bfloat16

            assert k_cache_4d.dim()==4
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
            # # 获取当前序列最新的那个 slot 索引
            # current_slot = context.slot_mapping[0].item() 
            # # 查验这个 slot 里的最大值
            # print(f"DEBUG: Max value in newly assigned slot {current_slot}: {self.k_cache[current_slot].max().item()}")
            o,_=flash_mla_with_kvcache(
                q_4d,
                k_cache_4d,
                block_tables,
                cache_seqlens,
                dv,
                tile_scheduler_metadata,
                num_splits,
                softmax_scale=self.scale,
                causal=True
            )
            
        return o.squeeze(1)



