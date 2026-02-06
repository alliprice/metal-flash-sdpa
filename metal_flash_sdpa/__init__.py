"""Metal Flash Attention for PyTorch MPS — drop-in replacement for F.scaled_dot_product_attention."""

import torch
import torch.nn.functional as F
from metal_flash_sdpa._C import mfa_attention_forward

_original_sdpa = F.scaled_dot_product_attention


class MetalFlashAttentionForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale):
        # PyTorch SDPA format: [B, Hq, R, D] -> MFA format: [B, R, Hq, D]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        o = mfa_attention_forward(q, k, v, scale)

        # MFA format: [B, R, Hq, D] -> PyTorch SDPA format: [B, Hq, R, D]
        return o.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is Phase 3")


def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
    """Drop-in replacement for F.scaled_dot_product_attention that uses Metal Flash Attention on MPS."""
    if (query.device.type == 'mps'
        and dropout_p == 0.0
        and attn_mask is None
        and not is_causal
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.is_grad_enabled()):
        s = scale if scale is not None else query.size(-1) ** -0.5
        return MetalFlashAttentionForward.apply(query, key, value, s)
    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, enable_gqa=enable_gqa)


def enable():
    """Monkey-patch F.scaled_dot_product_attention to use Metal Flash Attention on MPS."""
    F.scaled_dot_product_attention = patched_sdpa


def disable():
    """Restore original SDPA."""
    F.scaled_dot_product_attention = _original_sdpa
