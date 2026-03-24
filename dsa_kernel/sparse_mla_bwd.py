# ruff: noqa
import tilelang
from tilelang import language as T
import torch
from .index import prepare_token_indices

#preprocess 计算delta，一个中间量，先预先算出来
#bwd 核心的backward kernel
#postprocess 转换数据类型的kernel
#sparse_mla_bwd 封装/调度函数
@tilelang.jit(out_idx=[-1])
def preprocess(
    H,  #Head数
    D,  #输出维度
    block_ND=32,    #分块大小
    num_stages=5,   #pipline stage数
    dtype="bfloat16",   #输入的O和dO的dtype
    accum_dtype="float",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float"

    S = T.symbolic('S') #运行时才知道的token总数

    shape = [S, H, D]   #O和dO的形状

    @T.prim_func
    def preprocess_kernel( 
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND)) as (bx, by):    # 2D网络    x维为每个block负责一个head  y维为token按32一块划分 也就是一个block负责一个head和一段token
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype) #分配临时变量到高速缓存中
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)    #把acc清零，从0累积
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):    #沿着D维分块循环 第k个D-block
                T.copy(O[by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND], o)
                T.copy(dO[by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND],
                       do)  #从全局内存加载O/dO的tile到fragment
                for i, j in T.Parallel(block_ND, block_ND):     #计算O*dO,累加到acc #内层循环：32个token 32个维度 小tile块
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1) #对d维求和，得到每个token的delta
            T.copy(delta, Delta[by * block_ND:(by + 1) * block_ND, bx]) #写回到全局delta

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(    #转换数据类型
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype="bfloat16",
    accum_dtype="float",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    S_kv = T.symbolic('S_kv')   #kv的token数

    dkv_shape = [S_kv, kv_group, D + D_tail]    #形状

    @T.prim_func
    def postprocess_kernel(
            dKV: T.Tensor(dkv_shape, accum_dtype),  #输入
            dKV_out: T.Tensor(dkv_shape, dtype),    #输出
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, threads=threads) as (bx, by): #按token分块 按group分块
            T.copy(
                dKV[bx * block_N:(bx + 1) * block_N, by, :],
                dKV_out[bx * block_N:(bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def bwd(
    H,  #Q的head数
    D,  #主维度 
    D_tail, #尾维度
    topk,
    kv_group=1, #KV分组数
    sm_scale=None,  #缩放因子
    is_causal=True,
    block_size=32,  #处理topk的分块大小
    num_stages=0,
    threads=128,    
    indices_dtype="int32",
    dtype="bfloat16",
    accum_dtype="float",
):
    assert is_causal == True, 'non-casual is not supported now'
    assert topk % block_size == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'    #强制topk整除block_size
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    assert indices_dtype == "int32"

    if sm_scale is None:
        sm_scale = (D + D_tail)**(-0.5) #默认缩放因子

    B_plus_one = T.symbolic('B_plus_one')   #offsets的长度
    S = T.symbolic('S') #packed后的token总数

    H_kv = H // kv_group    #每个kvgroup对应的head数
    q_shape = [S, H, D + D_tail]    #输入的形状
    k_shape = [S, kv_group, D + D_tail]
    o_shape = [S, H, D] #输出形状
    indices_shape = [S, kv_group, topk]#每个token   每个kvgroup topk个索引  指向该样本内的key位置
    delta_shape = [S, H]
    lse_shape = [S, H]
    offsets_shape = [B_plus_one]
    token_indices_shape = [S, 2]
    assert indices_dtype == "int32"
    assert dtype == "bfloat16"
    assert accum_dtype == "float"

    H = H_kv    #H改成kv 对应head数
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16) #把head数padding到2的幂
    BS = block_size #换个别名
    NS = tilelang.cdiv(topk, block_size)    #需要多少轮才能把topk个key处理完

    split_store = 2 #写回拆成两次做

    @T.prim_func
    def sparse_mla_bwd_kernel(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(k_shape, dtype),
            dO: T.Tensor(o_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),    #每个query的topk索引
            Lse: T.Tensor(lse_shape, accum_dtype),
            Delta: T.Tensor(delta_shape, accum_dtype),
            Offsets: T.Tensor(offsets_shape, indices_dtype),
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype), #映射token->样本，样本内位置
            dQ: T.Tensor(q_shape, dtype),   #输出
            dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, kv_group, threads=threads) as (b_s_i, bz): #x维 全局packedtoken下标    y维 第几个kvgroup   一个block对应一个token，一个kv group，一段head
            Q_shared = T.alloc_shared([padded_H, D], dtype) #分配shared memory
            Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([padded_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([padded_H, BS], dtype)   #概率p
            dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)  #dS，虽然这是dP
            dQ_shared = T.alloc_shared([padded_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            #寄存器变量分配，真正累加的地方
            acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([padded_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([padded_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            #写回优化
            acc_dkv_shared = T.view(KV_shared, shape=[BS // split_store, D], dtype=accum_dtype)
            acc_dkv_tail_shared = T.view(
                KV_tail_shared, shape=[BS // split_store, D_tail], dtype=accum_dtype)
            #当前token样本与位置
            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]   #在packed起始、结束位置

            max_kv_i = s_i  #mask
            #加载Q、dO
            T.copy(Q[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D], Q_shared)
            T.copy(Q[bos + s_i, bz * padded_H:(bz + 1) * padded_H, D:], Q_tail_shared)
            T.copy(dO[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D], dO_shared)
            #清零累加器
            T.clear(acc_dq)
            T.clear(acc_dq_tail)
            #优化
            T.annotate_layout({
                dQ_shared: tilelang.layout.make_swizzled_layout(dQ_shared),
                dQ_tail_shared: tilelang.layout.make_swizzled_layout(dQ_tail_shared),
            })

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):     #外层循环：按topk分块处理
                # Check which indices are valid
                for bi_i in T.Parallel(BS): #因果约束
                    mask[bi_i] = (Indices[bos + s_i, bz, i_i * BS + bi_i] <= max_kv_i) & (
                        Indices[bos + s_i, bz, i_i * BS + bi_i] != -1)
                #初始化acc_p
                # Compute attention scores
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))
                
                # Load KV, V for this block of indices  加载 KV
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i], bz,
                                              d_i]
                #计算logits
                T.gemm(
                    Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                #加载尾维
                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i],
                                                bz, D + d_i]
                T.gemm(
                    Q_tail_shared,
                    KV_tail_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol)
                #把logits变成概率   
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp(acc_p[h_i, bi_i] * sm_scale -
                                             Lse[bos + s_i, bz * padded_H + h_i])
                #存起来
                T.copy(acc_p, P_shared_cast)
                #计算dP
                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True)
                #把dP变成dS
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (
                        acc_dp[h_i, bi_i] - Delta[bos + s_i, bz * padded_H + h_i]) * sm_scale
                #存起来
                T.copy(acc_dp, dP_shared_cast)
                #用dS算dQ
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)
                #计算KV主维的acc_dkv
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True)
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol)
                #计算dK_tail
                T.clear(acc_dkv_tail)
                T.gemm(
                    dP_shared_cast,
                    Q_tail_shared,
                    acc_dkv_tail,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):    #写回拆成两半
                    for bi_i, d_i in T.Parallel(BS, D): #一半拷贝到写回缓冲视图
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i,
                                                d_i] = acc_dkv_tail[bi_i + s * (BS // split_store),
                                                                    d_i]
                    #写回主维dkv
                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s *
                                              (BS // split_store)], bz, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4])
                    #写回尾维dkv
                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s *
                                              (BS // split_store)], bz, D + d_i * 4],
                            acc_dkv_tail_shared[bi_i, d_i * 4])
            #dQ的写回不会冲突，直接写回即可
            #dKV 多个query会命中同一个key，需要原子累加
            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)   #拷回shared
            T.copy(acc_dq_tail, dQ_tail_shared)
            #写回全局
            T.copy(dQ_shared, dQ[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D])
            T.copy(dQ_tail_shared, dQ[bos + s_i, bz * padded_H:(bz + 1) * padded_H, D:])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd(q,
                   kv,
                   o,
                   do,
                   indices, #topk 索引
                   lse,
                   offsets,
                   sm_scale=None,
                   is_casual=True,
                   return_kernel=False,
                   delta=None):
    assert q.is_contiguous()#连续性检查
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    S, H, dim_plus_tail_dim = q.shape   #token总数 Head数 q的维度
    S_kv, kv_group, _ = kv.shape    #token总数 KV分组数
    assert kv.shape[-1] == dim_plus_tail_dim    #保证最后一维一致
    assert S == S_kv    #保证Q和KV的token数一致
    # dim should be assigned
    D = 512 #写死 输出主维为512维   

    D_tail = dim_plus_tail_dim - D  #计算尾维
    topk = indices.shape[-1]    #读取topk
    assert indices.shape == (S, kv_group, topk) #检查indices形状
    assert lse.shape == (S, H)  #检查lse形状

    token_indices = prepare_token_indices(offsets)  #token t属于第几个样本、在样本中的位置

    # Get kernels
    preprocess_kernel = preprocess(H, D)    #生成三个GPU kernel
    bwd_kernel = bwd(H, D, D_tail, topk, kv_group, sm_scale, is_casual)
    postprocess_kernel = postprocess(D, D_tail, kv_group)
    
    if delta is None:
        o = o.contiguous()
        do = do.contiguous()
        delta = preprocess_kernel(o, do)    #执行preprocess kernel
    dkv = torch.zeros_like(kv, dtype=torch.float32) #分配
    dq = bwd_kernel(q, kv, do, indices, lse, delta, offsets, token_indices, dkv)    #调用主kernel
    dkv = postprocess_kernel(dkv)   #后处理

    return dq, dkv      #返回

