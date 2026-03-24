# Modified from: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
import torch
import torch.nn.functional as F
import functools
from typing import Callable, Any


def tensor_cache(fn: Callable[..., torch.Tensor],) -> Callable[..., torch.Tensor]:  #缓存最近一次调用该函数的输入和输出，如果下一次调用输入和上一次一样，就直接返回上一次的输出，不用重新计算。
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.  


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  
        nonlocal last_args, last_kwargs, last_result

        if (last_args is not None and last_kwargs is not None) and \
            (len(args) == len(last_args) and len(kwargs) == len(last_kwargs)) and \
                all(a is b for a, b in zip(args, last_args, strict=False)) and \
                    all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
            return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor: #从前缀和数组计算每个序列的长度
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_cu_seqlens_from_lens(   #从每个序列的长度计算前缀和数组，前面加一个0
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_lens_from_cu_seqlens(cu_seqlens: torch.LongTensor,) -> torch.LongTensor:    #好像重复了
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])  #每个序列长度L0，L1，L2······，然后生成0~L0-1，0~L1-1，0~L2-1······的position id，并拼接成一个张量返回


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor: #从前缀和数组计算每个位置所在的序列id，序列id从0开始，就是batch id
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1         


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:   #合成二维索引，第一维是序列id，第二维是位置id
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)

@tensor_cache
def prepare_cu_seqlens_from_position_ids(   #用position id计算cu_seqlens
) -> torch.LongTensor:
    starts = (position_ids == 0).nonzero(as_tuple=True)[0]
    total_len = position_ids.new_tensor([position_ids.numel()])
    boundaries = torch.cat([starts, total_len])
    lens = torch.diff(boundaries)
    cu_seqlens = prepare_cu_seqlens_from_lens(lens, dtype=dtype)
    return cu_seqlens