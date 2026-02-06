"""Metal Flash Attention for PyTorch MPS — drop-in replacement for F.scaled_dot_product_attention."""

import torch
import torch.nn.functional as F
from metal_flash_sdpa._C import mfa_attention_forward, mfa_attention_backward

_original_sdpa = F.scaled_dot_product_attention

MIN_SEQ_LEN = 256  # MFA overhead exceeds benefit below this threshold

# Dispatch counter for diagnostics (incremented each time MFA is used)
_dispatch_count = 0


class MetalFlashAttentionForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale, is_causal):
        # PyTorch SDPA format: [B, Hq, R, D] -> MFA format: [B, R, Hq, D]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        o, lse = mfa_attention_forward(q, k, v, scale, is_causal)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        ctx.is_causal = is_causal

        # MFA format: [B, R, Hq, D] -> PyTorch SDPA format: [B, Hq, R, D]
        return o.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse = ctx.saved_tensors

        # PyTorch SDPA format: [B, Hq, R, D] -> MFA format: [B, R, Hq, D]
        d_o = grad_output.transpose(1, 2).contiguous()

        dq, dk, dv = mfa_attention_backward(q, k, v, o, lse, d_o, ctx.scale, ctx.is_causal)

        # MFA format: [B, R, Hq, D] -> PyTorch SDPA format: [B, Hq, R, D]
        # dK/dV are [B, C, Hq, D] from MFA — transpose to [B, Hq, C, D]
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None


_debug_log = False
_fallback_count = 0

# Cache for trivial mask check: avoid repeated GPU syncs for the same mask tensor
_last_mask_ptr = None
_last_mask_trivial = False


def _is_trivial_mask(attn_mask):
    """Check if attention mask is all-True (no actual masking needed).

    Caches result by data pointer so repeated calls with the same mask tensor
    (e.g., 60 transformer blocks sharing one mask) only sync once.
    """
    global _last_mask_ptr, _last_mask_trivial
    if attn_mask is None:
        return True
    if attn_mask.dtype != torch.bool:
        return False
    ptr = attn_mask.data_ptr()
    if ptr == _last_mask_ptr:
        return _last_mask_trivial
    result = attn_mask.all().item()
    _last_mask_ptr = ptr
    _last_mask_trivial = result
    return result


def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
    """Drop-in replacement for F.scaled_dot_product_attention that uses Metal Flash Attention on MPS."""
    global _dispatch_count, _fallback_count, _debug_log
    if (query.device.type == 'mps'
        and dropout_p == 0.0
        and _is_trivial_mask(attn_mask)
        and query.size(-2) >= MIN_SEQ_LEN
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        _dispatch_count += 1
        s = scale if scale is not None else query.size(-1) ** -0.5
        return MetalFlashAttentionForward.apply(query, key, value, s, is_causal)
    _fallback_count += 1
    if _debug_log and _fallback_count <= 50:
        reasons = []
        if query.device.type != 'mps':
            reasons.append(f"device={query.device}")
        if dropout_p != 0.0:
            reasons.append(f"dropout={dropout_p}")
        if attn_mask is not None and not _is_trivial_mask(attn_mask):
            reasons.append(f"attn_mask={attn_mask.shape} (has False values)")
        if query.size(-2) < MIN_SEQ_LEN:
            reasons.append(f"seq_len={query.size(-2)}<{MIN_SEQ_LEN}")
        if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            reasons.append(f"dtype={query.dtype}")
        print(f"  [MFA fallback #{_fallback_count}] Q={list(query.shape)} {query.dtype} | {', '.join(reasons)}")
    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, enable_gqa=enable_gqa)


def enable():
    """Monkey-patch F.scaled_dot_product_attention to use Metal Flash Attention on MPS."""
    F.scaled_dot_product_attention = patched_sdpa


def disable():
    """Restore original SDPA."""
    F.scaled_dot_product_attention = _original_sdpa


def get_dispatch_count():
    """Return the number of times MFA has been dispatched since last reset."""
    return _dispatch_count


def reset_dispatch_count():
    """Reset the MFA dispatch counter to zero."""
    global _dispatch_count
    _dispatch_count = 0
