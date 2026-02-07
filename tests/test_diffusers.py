"""Diffusers integration tests — skipped if diffusers is not installed."""
import pytest
import torch
import torch.nn.functional as F
import metal_flash_sdpa
from unittest.mock import patch

diffusers = pytest.importorskip("diffusers")


def test_diffusers_dispatch_intercept():
    """metal_flash_sdpa.enable() intercepts diffusers' native attention backend."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 8, 512, 64
    query = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)
    key = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)
    value = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)

    metal_flash_sdpa.disable()
    out_default = dispatch_attention_fn(query, key, value)

    metal_flash_sdpa.enable()
    with patch("metal_flash_sdpa.mfa_attention_forward",
               wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        out_mfa = dispatch_attention_fn(query, key, value)
        assert mock_fwd.called, "MFA forward was NOT called through diffusers dispatch!"

    max_diff = (out_mfa - out_default).abs().max().item()
    assert max_diff < 0.01, f"Output mismatch: {max_diff}"


def test_diffusers_attention_processor():
    """Test with diffusers Attention module (as used by transformer blocks)."""
    from diffusers.models.attention_processor import Attention

    D, H = 64, 8
    attn = Attention(query_dim=D * H, heads=H, dim_head=D).to(device="mps", dtype=torch.float16)

    B, R = 1, 512
    hidden_states = torch.randn(B, R, D * H, device="mps", dtype=torch.float16)

    metal_flash_sdpa.disable()
    with torch.no_grad():
        out_default = attn(hidden_states)

    metal_flash_sdpa.enable()
    with patch("metal_flash_sdpa.mfa_attention_forward",
               wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        with torch.no_grad():
            out_mfa = attn(hidden_states)
        assert mock_fwd.called, "MFA forward was NOT called through Attention processor!"

    max_diff = (out_mfa - out_default).abs().max().item()
    assert max_diff < 0.05, f"Output mismatch: {max_diff}"


def test_backward_through_diffusers():
    """Gradients flow through diffusers dispatch with MFA enabled."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 4, 512, 64
    query = torch.randn(B, R, H, D, device="mps", dtype=torch.float16, requires_grad=True)
    key = torch.randn(B, R, H, D, device="mps", dtype=torch.float16, requires_grad=True)
    value = torch.randn(B, R, H, D, device="mps", dtype=torch.float16, requires_grad=True)

    metal_flash_sdpa.enable()
    out = dispatch_attention_fn(query, key, value)
    loss = out.sum()
    loss.backward()

    assert query.grad is not None, "query.grad is None"
    assert key.grad is not None, "key.grad is None"
    assert value.grad is not None, "value.grad is None"
    assert query.grad.abs().sum() > 0, "query.grad is all zeros"


def test_small_seq_fallback_in_diffusers():
    """Small sequences fall back to original SDPA even through diffusers."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 4, 64, 64  # R=64 < MIN_SEQ_LEN=256
    query = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)
    key = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)
    value = torch.randn(B, R, H, D, device="mps", dtype=torch.float16)

    metal_flash_sdpa.enable()
    with patch("metal_flash_sdpa.mfa_attention_forward") as mock_fwd:
        with torch.no_grad():
            dispatch_attention_fn(query, key, value)
        assert not mock_fwd.called, "MFA should NOT be called for small sequences"


def test_mfa_call_counter():
    """MFA is called at least once during a forward+backward pass."""
    call_count = 0
    original_fwd = metal_flash_sdpa.mfa_attention_forward

    def counting_fwd(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_fwd(*args, **kwargs)

    metal_flash_sdpa.enable()
    import metal_flash_sdpa._C
    orig = metal_flash_sdpa._C.mfa_attention_forward
    metal_flash_sdpa._C.mfa_attention_forward = counting_fwd
    metal_flash_sdpa.mfa_attention_forward = counting_fwd

    try:
        B, H, R, D = 1, 8, 1024, 64
        Q = torch.randn(B, H, R, D, device="mps", dtype=torch.float16, requires_grad=True)
        K = torch.randn(B, H, R, D, device="mps", dtype=torch.float16, requires_grad=True)
        V = torch.randn(B, H, R, D, device="mps", dtype=torch.float16, requires_grad=True)

        out = F.scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
    finally:
        metal_flash_sdpa._C.mfa_attention_forward = orig
        metal_flash_sdpa.mfa_attention_forward = orig

    assert call_count >= 1, f"Expected at least 1 MFA call, got {call_count}"
