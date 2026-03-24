from typing import Optional
import torch
import torch.nn.functional as F
from .indexer_topk_reducesum import indexer_topk_reducesum_interface    #stage2 索引器前向并选top-k √
from .indexer_bwd import indexer_bwd_interface  #stage2 索引器的反向    √
from .full_indexer_bwd import full_indexer_bwd_interface    #stage1 索引器反向  √
from .sparse_mla_fwd import sparse_mla_fwd_interface    #稀疏注意力 stage2语言模型的注意力前向计算，直接算出O矩阵   √
from .sparse_mla_bwd import sparse_mla_bwd      #稀疏注意力 stage2反向  √
from .sparse_mla_topk_reducesum import sparse_mla_topk_reducesum_interface  #stage2 bwd中再算一次注意力，给索引器训练提供注意力矩阵
from .dense_mla_fwd import dense_mla_fwd_interface  #stage1 稠密注意力         √  √ √
from einops import einsum, repeat

class DSAFunction(torch.autograd.Function): #语言模型

    @staticmethod
    def forward(    #前向
        ctx,    #存储forward结果
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        # topk_indices, index_score = ref_index_score(index_q, weights, index_k, topk)
        topk_indices, index_score = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)  #索引器打分并选择top-k
        o, lse = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)   #只在top-k上算注意力输出
        ctx.save_for_backward(q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets) #把backward要用到的存起来
        ctx.topk = topk
        ctx.dim_v = dim_v
        ctx.sm_scale = sm_scale
        return o, topk_indices  

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,   #对o的上游梯度
        _1: torch.Tensor,
    ):
        q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets = ctx.saved_tensors    #取回张量
        attn_score = sparse_mla_topk_reducesum_interface(
            q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), lse, offsets,
            dim_v=ctx.dim_v).squeeze(-2)    #算注意力 索引器反向需要
        dq, dkv = sparse_mla_bwd(   #稀疏注意力反向
            q,
            kv.unsqueeze(-2),
            o,
            do,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            sm_scale=ctx.sm_scale)
        dindex_q, dweights, dindex_k = indexer_bwd_interface(index_q, weights, index_k, attn_score,
                                                             index_score, topk_indices, offsets)
        return dq, dkv.squeeze(-2), dindex_q, dindex_k, dweights, None, None, None, None    #与forward输入对应，最后四个的梯度不需要
        #return dq, dkv.squeeze(-2), None, None, None, None, None, None, None    #索引器的不要


def deepseek_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunction.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)


class DSAFunctionWarmup(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        o, lse = dense_mla_fwd_interface(q, kv.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)    #稠密注意力
        ctx.save_for_backward(q, kv, index_q, index_k, weights, offsets)    #存起来
        return o

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor
    ):
        q, kv, index_q, index_k, weights, offsets = ctx.saved_tensors   #取出来

        dindex_q, dweights, dindex_k = full_indexer_bwd_interface(q, kv, index_q, weights, index_k, offsets)    #稠密注意力 索引器反向

        return None, None, dindex_q, dindex_k, dweights, None, None, None, None #只对索引器更新


def deepseek_sparse_attention_warmup(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunctionWarmup.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)
#packed index 实现过程，几个index变量的含义     √
#在线softmax
#各网格grid
#原子相加
#topk映射   √

#tilelang共性

