import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class YoutuMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class YoutuAttention(nn.Module):
    """MLA (Multi-Latent Attention) with weight absorption for nanovllm.

    Absorb strategy:
      - Key absorb:   q_absorbed = q_nope @ W_UK^T  (W_UK from kv_b_proj's key part)
        so that score = [q_absorbed; q_rope] @ [c_kv; k_rope]^T
      - Value absorb: o_proj_absorbed = W_DV @ o_proj  (W_DV from kv_b_proj's value part)
        so attn output is projected through the absorbed o_proj directly

    KV cache stores the compressed representation replicated across all Q heads:
      - K cache: [c_kv(kv_lora_rank); k_rope(qk_rope_head_dim)] with H KV heads
      - V cache: [c_kv(kv_lora_rank); zeros(qk_rope_head_dim)] with H KV heads
      padded so K and V have the same head_dim for flash_attn compatibility

    Cache config per TP rank:
      num_kv_heads = num_attention_heads // tp_size
      head_dim     = kv_lora_rank + qk_rope_head_dim
    """

    @staticmethod
    def get_cache_config(hf_config, tp_size):
        """Return (num_kv_heads_per_tp, head_dim) for KV cache allocation."""
        num_kv_heads = hf_config.num_attention_heads // tp_size  # MHA: each TP rank holds its own heads
        head_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
        return num_kv_heads, head_dim

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int | None,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.k_head_dim = kv_lora_rank + qk_rope_head_dim  # compressed key head dim

        self.scale = self.qk_head_dim ** -0.5

        # ===================== Q projection =====================
        if q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                hidden_size,
                q_lora_rank,
                bias=False,
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )

        # ===================== KV compression =====================
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)

        # kv_b_proj: full (not TP-sharded), used only for absorb init
        self.kv_b_proj = ReplicatedLinear(
            kv_lora_rank,
            self.total_num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # ===================== RoPE =====================
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # ===================== Attention =====================
        # num_kv_heads=1: single compressed KV head, Q has self.num_heads
        self.attn = Attention(
            self.num_heads,
            self.k_head_dim,
            self.scale,
            self.num_heads,
        )

        # ===================== Output projection =====================
        self.o_proj = RowParallelLinear(
            self.total_num_heads * v_head_dim,
            hidden_size,
            bias=False,
        )

        # Cached absorbed weights (lazy init after weight loading)
        self._w_key: torch.Tensor | None = None     # [H, kv_lora_rank, qk_nope_head_dim]
        self._w_vo: torch.Tensor | None = None       # [H, kv_lora_rank, v_head_dim]

    def _init_absorbed_weights(self):
        """Extract and cache absorbed weights from kv_b_proj.

        W_kv_b: [total_num_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank]
          - W_UK = W_kv_b[:, :qk_nope_head_dim, :]   -> for key absorption
          - W_UV = W_kv_b[:, qk_nope_head_dim:, :]    -> for value absorption
        """
        tp_rank = dist.get_rank()
        W = self.kv_b_proj.weight  # [total_num_heads * (nope + v), kv_lora_rank]
        W_kv_b = W.view(
            self.total_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        head_start = tp_rank * self.num_heads
        head_end = head_start + self.num_heads

        # W_UK: [H, nope, r] -> for q_nope @ W_UK => q_absorbed [H, r]
        self._w_key = W_kv_b[head_start:head_end, :self.qk_nope_head_dim, :].contiguous()

        # W_UV: [H, v, r] -> for (attn_weights @ c_kv) @ W_UV^T => [H, v]
        self._w_vo = W_kv_b[head_start:head_end, self.qk_nope_head_dim:, :].contiguous()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [N, hidden_size]

        # Lazy init absorbed weights
        if self._w_key is None:
            self._init_absorbed_weights()

        # ===================== Q =====================
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)

        q = q.view(-1, self.num_heads, self.qk_head_dim)  # [N, H, qk_head_dim]
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
  
        # ===================== KV compression =====================
        kv_latent = self.kv_a_proj_with_mqa(hidden_states)  # [N, kv_lora_rank + rope]
        kv_c, k_rope = torch.split(
            kv_latent, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c)  # [N, kv_lora_rank]
        q_nope_t = q_nope.transpose(0, 1)  # [H, N, nope]
        q_absorbed = torch.bmm(q_nope_t, self._w_key)  # [H, N, r]
        q_absorbed = q_absorbed.transpose(0, 1)  # [N, H, r]
        k_rope = k_rope.unsqueeze(1)  # [N, 1, rope]
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
        q_states = torch.cat([q_absorbed, q_rope], dim=-1)

        k_states = torch.cat([kv_c_normed.unsqueeze(1), k_rope], dim=-1)
        k_states = k_states.expand(-1, self.num_heads, -1)  # [N, H, k_head_dim]
        v_states = F.pad(
            kv_c_normed.unsqueeze(1),
            [0, self.qk_rope_head_dim],
        )  # [N, 1, k_head_dim]
        v_states = v_states.expand(-1, self.num_heads, -1)  # [N, H, k_head_dim]

        o = self.attn(q_states, k_states, v_states)  # [N, H, k_head_dim]
        o = o[..., :self.kv_lora_rank]  # [N, H, kv_lora_rank]
        o_t = o.transpose(0, 1)  # [H, N, r]
        o_v = torch.bmm(o_t, self._w_vo.transpose(1, 2))  # [H, N, v]
        o_v = o_v.transpose(0, 1)  # [N, H, v]

        # ===================== Output projection =====================
        output = self.o_proj(o_v.flatten(1, -1))  # [N, hidden_size]
        return output


class YoutuDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.self_attn = YoutuAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            max_position=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = YoutuMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # smoke test: 跳过最后的 MLP / MoE
        return hidden_states, residual


class YoutuModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [YoutuDecoderLayer(config) for _ in range(1)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        hidden_states, residual = self.layers[0](positions, hidden_states, residual)

        print("layer0 hidden_states shape:", hidden_states.shape)
        print("layer0 hidden_states first token first 10 dims:", hidden_states[0, :10])

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class YoutuForCausalLM(nn.Module):
    # Weight mapping: HF -> nanovllm for MLP merged projections
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Direct name mapping: HF MLA weight names -> nanovllm names

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.model = YoutuModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
