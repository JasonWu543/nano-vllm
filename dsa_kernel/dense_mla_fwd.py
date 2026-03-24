# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from .index import prepare_token_indices


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dense_mla_fwd(
    heads,
    dim,
    tail_dim,
    kv_group=1,
    sm_scale=None,  #softmax scale
    is_causal=True,
    CP0=True,
    block_I=32,
    num_stages=2,
    threads=128,
):
#约束 默认值
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}" #要求都是2的幂
    assert is_causal == True, "non-casual is not supported"     #只支持casual attention
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5    #没有的话就用标准的缩放因子
    else:
        sm_scale = sm_scale

    batch_plus_one = T.symbolic("batch_plus_one")   #offsets长度
    seq_len = T.symbolic("seq_len") #总token数
 #定义各张量的形状
    head_kv = heads // kv_group #定义形状
    q_shape = [seq_len, heads, dim + tail_dim]
    kv_shape = [seq_len, kv_group, dim + tail_dim]
    o_shape = [seq_len, heads, dim] #输出形状
    lse_shape = [seq_len, heads]    #lse形状
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    indices_dtype = "int32" #数据类型
    dtype = "bfloat16"
    accum_dtype = "float"
#处理head padding
    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)  #把head padding到2的幂，至少为16
    if padded_H != H:   #如果padding了，只允许kv_group为1
        assert (
            kv_group == 1
        ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    BI = block_I
    D = dim
    D_tail = tail_dim

    if head_kv > 64:    #如果大于64，就进行切块，拆成每次64个的复制快处理
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64  #每个block处理多少head

    @T.prim_func
    def main(   #GPU kernel
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
           Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(  #定义kernel的并行方式   thread block
                seq_len * REPLICATE_H, kv_group, threads=threads) as (
                    bx, #x方向：第几个token
                    by,#y方向：第几个KV group
                ):
    #分配临时存储
        #alloc_shared：共享内存
        #alloc_fragment:寄存器  
            #分配到GPU上的临时存储
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)#把当前query的Q，KV block的K搬到shared
            mask = T.alloc_fragment([BI], "bool")
            #softmax在线计算需要的变量
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype) #累计输出（还没softmax，已exp)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)    #当前block的score，未exp
            S_shared = T.alloc_shared([H_per_block, BI], dtype) #把acc_s转换数据类型再放入shared memory，一份拷贝
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)   #softmax分母的累积值
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype) #当前block的分母增量
            alpha = T.alloc_fragment([H_per_block], accum_dtype)    #max更新后，之前的累积部分的缩放因子
            m_i = T.alloc_fragment([H_per_block], accum_dtype)  #未softmax之前看到的最大得分
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype) #上一轮block结束时的最大值
        #初始化
            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H) #当前block计算对应的全局token编号
            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]   #batch id和 这个token在序列里的位置
            bos, eos = Offsets[b_i], Offsets[b_i + 1]   #offsets 长度为batch+1 通过offsets累计量计算该b_i在packed token 里的起始下标和结束下标  而每个token在packed里的下标是bos+s_i
            g_i = by    # 第几个kv group
            q_i = s_i   #query在序列里的位置
            max_kv_i = q_i  # for causal mask: can only attend to positions <= current position #最大只能看到自己
            
            # Number of KV blocks to iterate (for dense attention, iterate all KV up to current position)
            NI = tilelang.cdiv(max_kv_i + 1, BI)    #向上取整 ceil（a/b），需要遍历多少个kv block

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block   #当前block对应KV group的某一段head索引范围

            T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
            T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)  #把当前这一段范围的Q拷贝到共享内存

            for i_i in T.Pipelined(NI, num_stages=num_stages):      #进入循环 KV block 对kv的分块
                # Compute the starting KV index for this block
                kv_start = i_i * BI #当前块负责的key的索引范围起始

                for bi_i in T.Parallel(BI): #块内的偏移
                    # Causal mask: only attend to valid positions
                    mask[bi_i] = (kv_start + bi_i) <= max_kv_i  #计算该位置key是否合法（不能看到未来）

                for bi_i, d_i in T.Parallel(BI, D): #加载主维度到KV shared
                    kv_idx = T.min(kv_start + bi_i, max_kv_i)
                    KV_shared[bi_i, d_i] = KV[bos + kv_idx, g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):    #加载尾维度到KV shared
                    kv_idx = T.min(kv_start + bi_i, max_kv_i)
                    K_tail_shared[bi_i, d_i] = KV[bos + kv_idx, g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):   #初始化本块 logits
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm( #主维乘法
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm( #尾维乘法
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)   #保留旧max
                T.reduce_max(acc_s, m_i, dim=1, clear=False)    #更新max
                for h_i in T.Parallel(H_per_block): #计算alpha
                    alpha[h_i] = T.exp((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):   #softmax分子 对分子exp
                    acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?  #本块的key对分母的贡献
                for h_i in T.Parallel(H_per_block): #更新全局分母
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D): #旧的分子进行缩放
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared) #本块权重存起来
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow) #计算输出分子O（还没除以softmax分母）

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D): #输出归一化
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block): #计算LSE
                sumexp[h_i] = T.log(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[bos + s_i, H0:H1, :])  #写回全局输出
            T.copy(sumexp, Lse[bos + s_i, H0:H1])

    return main


def dense_mla_fwd_interface(q,
                             kv,
                             offsets,
                             sm_scale=None,
                             return_p_sum: bool = False,
                             d_v=512,
                             block_I=32,
                             num_stages=2,
                             threads=128):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous()
    seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    assert seq_len == seq_len_kv

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim

    token_indices = prepare_token_indices(offsets)

    kernel = dense_mla_fwd(
        heads,
        dim,
        tail_dim,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads)
    out, lse = kernel(q, kv, offsets, token_indices)
    return out, lse