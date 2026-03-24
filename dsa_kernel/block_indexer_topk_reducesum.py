import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang
import tilelang.language as T
from typing import Optional

from util import get_abs_err, get_err_ratio
from index import prepare_token_indices

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def tl_gather_qk_reducesum_impl(
        heads,
        dim,
        num_candidates,
        sm_scale=None,
        block_I=32, #每轮处理BI个
        num_stages=2,
        threads=128,
):
    """
    Fused kernel: gather K by indices + QK matmul + ReLU + weighted reduce-sum.

    Args:
        heads: number of attention heads H
        dim: head dimension D
        num_candidates: total number of candidate tokens per query (block_topk * block_size)
        sm_scale: softmax scale factor (default: dim ** -0.5)
        block_I: tile size for iterating over candidates
        num_stages: pipeline stages
        threads: threads per block
    """
    if sm_scale is None:
        sm_scale = dim ** -0.5

    assert num_candidates % block_I == 0, \
        f"num_candidates ({num_candidates}) must be divisible by block_I ({block_I})"   #保证整除

    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    batch_plus_one = T.symbolic("batch_plus_one")

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(H), 16)    #补充到2的幂
    D = dim
    BI = block_I
    NI = num_candidates // block_I

    dtype = "bfloat16"
    accum_dtype = "float"
    indices_dtype = "int32"
    #张量形状
    q_shape = [seq_len, H, D]
    k_shape = [seq_len_kv, D]
    weights_shape = [seq_len, H]
    token_indices_shape = [seq_len, num_candidates]
    offsets_shape = [batch_plus_one]
    seq_token_indices_shape = [seq_len, 2]
    score_shape = [seq_len, num_candidates]

    @T.prim_func
    def gather_qk_reducesum_kernel(
            Q: T.Tensor(q_shape, dtype),  # [seq_len, H, D]
            K: T.Tensor(k_shape, dtype),  # [seq_len_kv, D]
            Weights: T.Tensor(weights_shape, dtype),  # [seq_len, H]
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # [seq_len, num_candidates]
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # [batch_plus_one]
            SeqTokenIndices: T.Tensor(seq_token_indices_shape, indices_dtype),  # [seq_len, 2]
            Score: T.Tensor(score_shape, accum_dtype),  # [seq_len, num_candidates]
    ):
        with T.Kernel(seq_len, threads=threads) as (bx,):   #seq_len个block，每个block一个query
            # Shared memory allocations
            #内存分配
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            K_shared = T.alloc_shared([BI, D], dtype)
            W_shared = T.alloc_fragment([padded_H], accum_dtype)

            # Fragment for accumulating QK scores: [padded_H, BI]
            acc_s = T.alloc_fragment([padded_H, BI], accum_dtype)
            # Fragment for weighted reduced score per block: [BI]
            reducesum = T.alloc_fragment([BI], accum_dtype)

            # Resolve batch index and sequence position
            b_i, s_i = SeqTokenIndices[bx, 0], SeqTokenIndices[bx, 1]   #根据全局token索引，找到对应的batch id和在batch内的序列位置 b_i恒为0
            bos = Offsets[b_i]  #当前样本在packed序列中的起始位置

            # Load Q[bos + s_i, :, :] into shared memory
                #加载Q
            T.copy(Q[bos + s_i, 0:padded_H, :D], Q_shared)

            # Load Weights[bos + s_i, :] into fragment
                #加载权重，pad的部分权重置0
            for h_i in T.Parallel(padded_H):
                if h_i < H:
                    W_shared[h_i] = Weights[bos + s_i, h_i]
                else:
                    W_shared[h_i] = 0

            # Iterate over candidate blocks
            for i_i in T.Pipelined(NI, num_stages=num_stages):  #每轮处理BI个候选
                # Gather K: load K[token_indices[bos+s_i, i_i*BI + bi_i]] into K_shared
                # token_indices == -1 means invalid, clamp to 0 for safe load
                for bi_i, d_i in T.Parallel(BI, D): #取出候选key的索引，加载到共享内存中
                    K_shared[bi_i, d_i] = K[
                        bos + T.max(TokenIndices[bos + s_i, i_i * BI + bi_i], 0),
                        d_i
                    ]

                # Initialize acc_s: set to 0 for valid indices, -inf for invalid (token_indices == -1)
                #初始化acc_s，合法位置为0，无效位置为-inf，确保后续matmul不受无效位置影响
                for h_i, bi_i in T.Parallel(padded_H, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        TokenIndices[bos + s_i, i_i * BI + bi_i] >= 0,
                        0,
                        -T.infinity(acc_s.dtype)
                    )

                # QK matmul: acc_s += Q_shared @ K_shared^T  -> [padded_H, BI]
                #计算QK得分，结果累加到acc_s中
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                #ReLU，缩放，加权
                # ReLU + scale + weighted reduce-sum over heads
                for h_i, bi_i in T.Parallel(padded_H, BI):
                    # ReLU
                    acc_s[h_i, bi_i] = T.max(acc_s[h_i, bi_i], 0)
                    # Multiply by weight and scale
                    acc_s[h_i, bi_i] = acc_s[h_i, bi_i] * W_shared[h_i] * sm_scale
                #对每个head求和
                # Reduce sum over heads: reducesum[bi] = sum_h acc_s[h, bi]
                T.reduce_sum(acc_s, reducesum, dim=0)
                #写回输出
                # Store result
                T.copy(reducesum, Score[bos + s_i, i_i * BI:i_i * BI + BI])

    return gather_qk_reducesum_kernel


def gather_qk_reducesum_interface(
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        token_indices: torch.Tensor,
        offsets: torch.Tensor,
        sm_scale: float = None,
        block_I: int = 32,
        num_stages: int = 2,
        threads: int = 128,
) -> torch.Tensor:
    """
    Fused gather-K + QK-matmul + ReLU + weighted-reducesum interface.

    Args:
        q: [total_seq_len, H, D] bf16 query tensor
        k: [total_seq_len, D] bf16 key tensor
        weights: [total_seq_len, H] bf16 weight tensor
        token_indices: [total_seq_len, num_candidates] int32 candidate token indices (-1 for invalid)
        offsets: [batch+1] int32 batch offsets
        sm_scale: softmax scale (default: D ** -0.5)
        block_I: tile size for candidates dimension
        num_stages: pipeline stages
        threads: threads per block

    Returns:
        score: [total_seq_len, num_candidates] float32 weighted scores
    """
    seq_len, H, D = q.shape #输入形状
    num_candidates = token_indices.shape[1] #每个query的候选token数量

    if sm_scale is None:
        sm_scale = D ** -0.5

    # Prepare sequence/token indices for the kernel
    seq_token_indices = prepare_token_indices(offsets)  #保持一致用法   单条序列

    # Ensure token_indices is int32
    if token_indices.dtype != torch.int32:
        token_indices = token_indices.to(torch.int32)

    # Allocate output
    score = torch.zeros(seq_len, num_candidates, dtype=torch.float32, device=q.device)  #输出张量，存放每个query对候选token的打分结果

    # Get compiled kernel   #编译kernel
    kernel = tl_gather_qk_reducesum_impl(   
        heads=H,
        dim=D,
        num_candidates=num_candidates,
        sm_scale=sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )

    # Run kernel    #执行kernel
    kernel(q, k, weights, token_indices, offsets, seq_token_indices, score)

    return score


def indexer_topk_reducesum_interface(
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        topk: int,
        offsets: torch.Tensor   #前缀和,
        block_size: int = 128,
        block_topk: int = 64,   #每个query选多少个block
        dtype: str = BF16,
        chunk_size: int = 2048,
):
    total_seq_len = q.shape[0]  #packed后的总token数
    device = q.device
    softmax_scale = q.shape[-1] ** -0.5

    all_topk_indices = torch.full((total_seq_len, topk), -1, dtype=torch.int64, device=device)  #创建两个输出张量
    all_topk_score = torch.full((total_seq_len, topk), float('-inf'), dtype=torch.float32, device=device)

    for batch_idx in range(offsets.shape[0] - 1):
        #还原出当前样本在packed张量中的范围
        start_idx = offsets[batch_idx].item()
        end_idx = offsets[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        #切出当前样本对应的q，weights，k
        q_batch = q[start_idx:end_idx]
        weights_batch = weights[start_idx:end_idx]
        k_batch = k[start_idx:end_idx]

        num_blocks = math.ceil(seq_len / block_size)
        pad_len = num_blocks * block_size - seq_len #按block size 分块，同时要pad多少token
        if pad_len > 0: #不能整除
            k_batch_padded = F.pad(k_batch, (0, 0, 0, pad_len), value=0.0)  # [num_blocks * block_size, D]  #对k pad
            block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=device)
            block_counts[-1] = block_size - pad_len  # last block has fewer tokens  #每个block里实际有多少token，最后一个block要减去pad的token数
        else:
            k_batch_padded = k_batch    
            block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=device)
        k_block_mean = (
                k_batch_padded.reshape(num_blocks, block_size, -1)
                .sum(dim=1)  # [num_blocks, D]
                / block_counts.unsqueeze(-1)
        ).to(k.dtype)  # [num_blocks, D]    #计算每个block的平均k向量

        for chunk_start in range(0, seq_len, chunk_size):   #分块处理query
            chunk_end = min(chunk_start + chunk_size, seq_len)  #结束位置
            chunk_len = chunk_end - chunk_start #chunk中的query数量

            q_chunk = q_batch[chunk_start:chunk_end]  # [chunk_len, H, D]   #切出来当前chunk的q
            weights_chunk = weights_batch[chunk_start:chunk_end]  # [chunk_len, H]
            chunk_end_block = math.ceil(chunk_end / block_size) #当前chunk覆盖到第几个block
            k_block_mean_visible = k_block_mean[:chunk_end_block]  # [chunk_end_block, D]   #当前chunk可见的block

            # Step 1: Compute block-level scores [chunk_len, chunk_end_block]   #对每个query和block，进行打分
            block_logits = einsum(q_chunk, k_block_mean_visible, 'cl h d, nb d -> cl h nb')
            block_logits = F.relu(block_logits)
            block_scores = (block_logits * weights_chunk.unsqueeze(-1)).sum(dim=-2,
                                                                            dtype=torch.float32) * softmax_scale  # [chunk_len, chunk_end_block]

            q_positions = torch.arange(chunk_start, chunk_end, device=device)  # [chunk_len]    #每个query在样本中的绝对位置，用于构造causal mask和mandatory block constraint
            block_starts = torch.arange(chunk_end_block, device=device) * block_size  # [chunk_end_block]   #每个block的起始token位置
            causal_block_mask = (q_positions.unsqueeze(1) >= block_starts.unsqueeze(0))  # [chunk_len, chunk_end_block]
            block_scores = torch.where(causal_block_mask, block_scores, torch.tensor(float('-inf'), device=device)) #非法block位置应用causal mask

            # Step 2: Block topk selection with mandatory constraint        #一些强制包含限制
            q_block_ids = q_positions // block_size #每个query所在的block id
            mandatory_last1 = q_block_ids   #强制包含自己这个block
            mandatory_last2 = (q_block_ids - 1).clamp(min=0)    #强制包含前一个block

            LARGE_SCORE = 1e9   #很大的分数，保证选中
            batch_indices = torch.arange(chunk_len, device=device)  #当前chunk内每个query的索引
            block_scores_for_topk = block_scores.clone()    #拷贝一份
            block_scores_for_topk[:, 0] = LARGE_SCORE   #强制block 0被所有query选中
            block_scores_for_topk[batch_indices, mandatory_last1] = LARGE_SCORE #强制选中自己block、上一个block
            block_scores_for_topk[batch_indices, mandatory_last2] = LARGE_SCORE

            actual_btopk = min(block_topk, chunk_end_block) #实际可选的topk
            _, selected_block_indices = torch.topk(block_scores_for_topk, k=actual_btopk,
                                                   dim=-1)  # [chunk_len, actual_btopk] #选topk个block，得到block id

            # Step 3: Expand selected blocks into token indices [chunk_len, block_topk * block_size]        #根据被选中的block id，构造候选token索引
            block_start_tokens = selected_block_indices * block_size  # [chunk_len, actual_btopk]   #每个被选中block的起始token位置
            offsets_in_block = torch.arange(block_size, device=device)  # [block_size]  #block内每个token的偏移
            candidate_indices = (block_start_tokens.unsqueeze(-1) + offsets_in_block).reshape(chunk_len,
                                                                                              -1)  # [chunk_len, actual_btopk * block_size] #每个query的候选token索引，包含被选中block内的所有token

            valid_mask = (candidate_indices <= q_positions.unsqueeze(1)) & (candidate_indices < seq_len)    #不能看未来token，除去pad
            token_indices = torch.where(valid_mask, candidate_indices, torch.tensor(-1, dtype=torch.int64,  #无效位置标记为-1
                                                                                    device=device))  # [chunk_len, actual_btopk * block_size]

            # Pad to full width if actual_btopk < block_topk
            if token_indices.shape[1] < block_topk * block_size:
                token_indices = F.pad(token_indices, (0, block_topk * block_size - token_indices.shape[1]), value=-1)

            # Step 4: Compute token-level scores for selected tokens (fused gather + QK + ReLU + reducesum) #细粒度计算打分
            num_candidates = token_indices.shape[1]  # block_topk * block_size  #每个query的候选token数量
            # Build per-chunk offsets for the fused kernel (single chunk = single "batch")
            chunk_offsets = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)  #把当前kernel当成一个batch序列
            token_scores = gather_qk_reducesum_interface(
                q_chunk, k_batch, weights_chunk,    #传入整个序列的k
                token_indices.to(torch.int32),  #k的索引列表
                chunk_offsets,
                sm_scale=softmax_scale,
                block_I=32,
                num_stages=2,
                threads=128,
            )  # [chunk_len, num_candidates]
            token_scores = torch.where(token_indices >= 0, token_scores, torch.tensor(float('-inf'), device=device))    #无效位置

            # Step 5: Topk token selection and store into all_topk_indices/all_topk_score
            actual_topk = min(topk, num_candidates) #实际可选的topk数量
            topk_scores, topk_local_ids = torch.topk(token_scores, k=actual_topk, dim=-1)  # [chunk_len, actual_topk]   #计算topk
            topk_token_indices = torch.gather(token_indices, dim=1, index=topk_local_ids)  # [chunk_len, actual_topk]   #根据local id从候选token索引中取出对应的全局token索引，注意有些可能是-1（无效）

            # Keep batch-local indices (consistent with ref_index_score)
            topk_global_indices = torch.where(topk_token_indices >= 0, topk_token_indices,
                                              torch.tensor(-1, dtype=torch.int64, device=device))       #序列内相对位置，无效位置仍然标记为-1
            topk_final_scores = F.softmax(topk_scores, dim=-1, dtype=torch.float32) #对topk分数进行softmax，得到概率分布
            topk_final_scores = torch.where(topk_token_indices >= 0, topk_final_scores,
                                            torch.tensor(float('-inf'), device=device))

            # Store results 
            global_start = start_idx + chunk_start  #当前chunk在整个序列中的起始位置
            global_end = start_idx + chunk_end  #当前chunk在整个序列中的结束位置
            if actual_topk < topk:  #部分写回，剩下保持初始值
                all_topk_indices[global_start:global_end, :actual_topk] = topk_global_indices
                all_topk_score[global_start:global_end, :actual_topk] = topk_final_scores
            else:   #整块写回
                all_topk_indices[global_start:global_end] = topk_global_indices
                all_topk_score[global_start:global_end] = topk_final_scores

    all_topk_indices = all_topk_indices.to(torch.int32)

    return all_topk_indices, all_topk_score
