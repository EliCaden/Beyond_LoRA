# models/gpt2_lm.py
from __future__ import annotations
from typing import Optional, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config

from models.lora_layers import LoRALinearAdapter


def _variant_to_repo(variant: str) -> str:
    """
    Map friendly variants to HF repos. 'tiny' -> a small public GPT-2.
    """
    alias = {
        "tiny": "sshleifer/tiny-gpt2",
        "small": "gpt2",
        "base": "gpt2",
        "gpt2": "gpt2",
    }
    return alias.get(variant, variant)


class _QKVLinearPack(nn.Module):
    """
    Replacement for GPT-2's attn.c_attn that returns cat([Q(x), K(x), V(x)], -1).
    Keeps q_proj and v_proj as separate Linear modules so LoRA can wrap them directly.
    """
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.is_qkv_pack = True  # marker for idempotency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1)


def _to_qkv_pack(c_attn: nn.Module) -> _QKVLinearPack:
    """
    Convert GPT-2's fused c_attn (Conv1D or Linear producing 3*H) into a pack of three Linear layers
    with weights/biases copied exactly so outputs are identical pre-LoRA.
    """
    if isinstance(c_attn, nn.Linear):
        # weight: [3H, H], bias: [3H]
        W = c_attn.weight.detach().clone()
        b = c_attn.bias.detach().clone() if c_attn.bias is not None else None
        H3, H = W.shape
        H1 = H3 // 3

        def mk(W_slice: torch.Tensor, b_slice: Optional[torch.Tensor]) -> nn.Linear:
            lin = nn.Linear(H, H1, bias=b is not None)
            with torch.no_grad():
                lin.weight.copy_(W_slice)
                if b_slice is not None:
                    lin.bias.copy_(b_slice)
            return lin

        q = mk(W[:H1, :], None if b is None else b[:H1])
        k = mk(W[H1:2 * H1, :], None if b is None else b[H1:2 * H1])
        v = mk(W[2 * H1:, :], None if b is None else b[2 * H1:])
        return _QKVLinearPack(q, k, v)

    # HF Conv1D case (older GPT-2): weight [H, 3H], bias [3H]
    W = c_attn.weight.detach().clone()
    b = getattr(c_attn, "bias", None)
    b = b.detach().clone() if b is not None else None
    H, H3 = W.shape
    H1 = H3 // 3

    def mk(W_slice: torch.Tensor, b_slice: Optional[torch.Tensor]) -> nn.Linear:
        lin = nn.Linear(H, H1, bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(W_slice.t().contiguous())
            if b_slice is not None:
                lin.bias.copy_(b_slice)
        return lin

    q = mk(W[:, :H1], None if b is None else b[:H1])
    k = mk(W[:, H1:2 * H1], None if b is None else b[H1:2 * H1])
    v = mk(W[:, 2 * H1:], None if b is None else b[2 * H1:])
    return _QKVLinearPack(q, k, v)


class GPT2CausalLM(nn.Module):
    """
    Wrapper around HF causal LM that:
      - builds a tokenizer (pads with eos if needed),
      - maps friendly variants to repos,
      - exposes add_lora_qv(r, alpha) to LoRA-wrap Q & V projections in every block.
    """
    def __init__(
        self,
        variant: str = "gpt2",
        device: Optional[torch.device | str] = None,
        pretrained: bool = True,
        **kwargs: Any,  # accept legacy/alias kwargs
    ):
        super().__init__()
        # alias: allow callers to pass backbone="gpt2" etc.
        backbone = kwargs.pop("backbone", None)
        use_id = backbone or variant
        repo = _variant_to_repo(use_id)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
        if pretrained:
            self.net = AutoModelForCausalLM.from_pretrained(repo)
        else:
            cfg = GPT2Config(
                n_layer=2, n_head=2, n_embd=128,
                n_positions=256, n_ctx=256, vocab_size=50257,
                bos_token_id=50256, eos_token_id=50256
            )
            self.net = AutoModelForCausalLM.from_config(cfg)

        self.net.resize_token_embeddings(len(self.tokenizer))
        if device is not None:
            self.net.to(device)

    def add_lora_qv(self, r: int, alpha: int) -> int:
        """
        Split fused c_attn into q/k/v Linear and wrap q & v with LoRA.
        Returns the number of adapters inserted (2 * n_layers).
        """
        inserted = 0
        # Expect GPT-2 architecture: transformer.h[*].attn.c_attn
        for block in self.net.transformer.h:
            attn = block.attn
            if not getattr(attn.c_attn, "is_qkv_pack", False):
                attn.c_attn = _to_qkv_pack(attn.c_attn)

            attn.c_attn.q_proj = LoRALinearAdapter(attn.c_attn.q_proj, r=r, alpha=alpha)
            attn.c_attn.v_proj = LoRALinearAdapter(attn.c_attn.v_proj, r=r, alpha=alpha)
            inserted += 2
        return inserted

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        return self.net(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.net.generate(*args, **kwargs)
    
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_heads(self):
        # GPT-2 ties lm_head <-> wte; unfreezing here will also unfreeze embeddings.
        for n, p in self.net.named_parameters():
            if "lm_head" in n:
                p.requires_grad = True


# Back-compat alias for scripts importing GPT2LMModel
GPT2LMModel = GPT2CausalLM

__all__ = ["GPT2CausalLM", "GPT2LMModel", "AutoTokenizer", "AutoModelForCausalLM"]
