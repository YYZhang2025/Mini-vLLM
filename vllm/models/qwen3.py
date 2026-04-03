import os
from glob import glob

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from transformers import Qwen3Config

from vllm.models.layers.activation import SiluAndMul
from vllm.models.layers.attention import Attention
from vllm.models.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from vllm.models.layers.layernorm import RMSNorm
from vllm.models.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.models.layers.rope import get_rope


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads

        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        assert num_kv_heads % tp_size == 0
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=None,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
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


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hf_config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            max_position=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            qkv_bias=getattr(hf_config, "attention_bias", True),
            head_dim=getattr(hf_config, "head_dim", None),
            rope_theta=getattr(hf_config, "rope_theta", 1000000),
            rope_scaling=getattr(hf_config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
        )
        self.input_layernorm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

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
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        hf_config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(hf_config) for _ in range(hf_config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, hf_config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(hf_config)
        self.lm_head = ParallelLMHead(hf_config.vocab_size, hf_config.hidden_size)
        if hf_config.tie_word_embeddings:
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


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


if __name__ == "__main__":
    import os

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, Qwen3Config

    # --------------------------------------------------
    # 1. init distributed for tensor-parallel modules
    # --------------------------------------------------
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    # --------------------------------------------------
    # 2. load tokenizer / config / model
    # --------------------------------------------------
    path = os.path.expanduser("~/model/Qwen3-0.6B/")

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    hf_config = Qwen3Config.from_pretrained(path)

    model = Qwen3ForCausalLM(hf_config).to(device)
    model.eval()

    print("Model loaded.")
    print(f"Device: {device}")

    # Optional: load local safetensors weights
    print("Loading weights...")
    load_model(model, path)
    print("Weights loaded.")

    # --------------------------------------------------
    # 3. build chat prompt
    # --------------------------------------------------
    prompt = "What is the Transformer?"
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\nPrompt:")
    print(prompt)

    # --------------------------------------------------
    # 4. tokenize
    # --------------------------------------------------
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [B, T]
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, T]

    print(f"\ninput_ids shape: {tuple(input_ids.shape)}")
    print(f"positions shape: {tuple(positions.shape)}")

    # --------------------------------------------------
    # 5. forward pass
    # --------------------------------------------------
    with torch.no_grad():
        hidden_states = model(input_ids, positions)  # [B, T, H]
        logits = model.compute_logits(hidden_states)  # [B, T, V]

    print(f"hidden_states shape: {tuple(hidden_states.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")

    assert hidden_states.shape == (batch_size, seq_len, hf_config.hidden_size)
    assert logits.shape == (batch_size, seq_len, hf_config.vocab_size)
    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(logits).all()

    # --------------------------------------------------
    # 6. greedy next-token prediction
    # --------------------------------------------------
    next_token_id = logits[:, -1, :].argmax(dim=-1)  # [B]
    next_token_text = tokenizer.decode(next_token_id.tolist(), skip_special_tokens=False)

    print("\nGreedy next token:")
    print("next_token_id:", next_token_id.tolist())
    print("next_token_text:", repr(next_token_text))

    # --------------------------------------------------
    # 7. optional: one-step append and decode
    # --------------------------------------------------
    # --------------------------------------------------
    # 7b. naive greedy decode
    # --------------------------------------------------
    max_new_tokens = 30
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        cur_seq_len = generated.shape[1]
        cur_positions = torch.arange(cur_seq_len, device=device, dtype=torch.long)
        cur_positions = cur_positions.unsqueeze(0).expand(generated.shape[0], -1)

        with torch.no_grad():
            hidden_states = model(generated, cur_positions)
            logits = model.compute_logits(hidden_states)

        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        generated = torch.cat([generated, next_token_id], dim=1)

        # stop if eos
        if tokenizer.eos_token_id is not None:
            if torch.all(next_token_id.squeeze(1) == tokenizer.eos_token_id):
                break

    print("\nNaive greedy generation:")
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=False))

    # --------------------------------------------------
    # 8. cleanup
    # --------------------------------------------------
    if dist.is_initialized():
        dist.destroy_process_group()
