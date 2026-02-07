"""Integration tests: SDPA monkey-patch plumbing, dispatch counter, masks, grad flow."""
import torch
import metal_flash_sdpa
from unittest.mock import patch


def test_grad_flow():
    """Gradients propagate through the patched SDPA."""
    B, H, R, D = 2, 4, 64, 64
    Q = torch.randn(B, H, R, D, device="mps", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, R, D, device="mps", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, R, D, device="mps", dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None, "Q.grad is None"
    assert K.grad is not None, "K.grad is None"
    assert V.grad is not None, "V.grad is None"
    assert Q.grad.abs().sum() > 0, "Q.grad is all zeros"
    assert K.grad.abs().sum() > 0, "K.grad is all zeros"
    assert V.grad.abs().sum() > 0, "V.grad is all zeros"


def test_min_seq_len_threshold():
    """Small sequences fall back to original SDPA; large sequences use MFA."""
    D = 64

    metal_flash_sdpa.enable()

    # Small sequence: should NOT call MFA
    Q_small = torch.randn(1, 4, 64, D, device="mps", dtype=torch.float16)
    K_small = torch.randn(1, 4, 64, D, device="mps", dtype=torch.float16)
    V_small = torch.randn(1, 4, 64, D, device="mps", dtype=torch.float16)

    with patch("metal_flash_sdpa.mfa_attention_forward") as mock_fwd:
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(Q_small, K_small, V_small)
        assert not mock_fwd.called, "MFA should NOT be called for small sequences"

    # Large sequence: SHOULD call MFA
    Q_large = torch.randn(1, 4, 512, D, device="mps", dtype=torch.float16)
    K_large = torch.randn(1, 4, 512, D, device="mps", dtype=torch.float16)
    V_large = torch.randn(1, 4, 512, D, device="mps", dtype=torch.float16)

    with patch("metal_flash_sdpa.mfa_attention_forward", wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(Q_large, K_large, V_large)
        assert mock_fwd.called, "MFA should be called for large sequences"


def test_dispatch_counter():
    """Dispatch counter increments correctly."""
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(5):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 5, f"Expected 5 dispatches, got {count}"


def test_trivial_mask_dispatches():
    """All-True boolean mask treated as no mask (dispatches to MFA)."""
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)

    all_true_mask = torch.ones(1, 512, dtype=torch.bool, device="mps")
    with torch.no_grad():
        out_masked = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=all_true_mask)
        out_none = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 2, f"Expected 2 dispatches with all-True mask, got {count}"
    diff = (out_masked - out_none).abs().max().item()
    assert diff < 1e-6, f"All-True mask output differs from None mask: {diff}"


def test_nontrivial_mask_falls_back():
    """Non-trivial mask falls back to original SDPA."""
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)

    mask_with_false = torch.ones(1, 512, dtype=torch.bool, device="mps")
    mask_with_false[0, 0] = False
    with torch.no_grad():
        torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask_with_false)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 0, f"Expected 0 dispatches with non-trivial mask, got {count}"


def test_training_loop():
    """Loss decreases over 10 training steps (smoke test)."""
    B, H, R, D = 1, 4, 32, 64
    target = torch.randn(B, H, R, D, device="mps", dtype=torch.float32)

    proj_in = torch.randn(D, D, device="mps", dtype=torch.float32, requires_grad=True)
    proj_out = torch.randn(D, D, device="mps", dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    losses = []
    lr = 0.001
    for step in range(10):
        x = torch.randn(B, H, R, D, device="mps", dtype=torch.float32)
        q = x @ proj_in
        k = x @ proj_in
        v = x @ proj_in
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out @ proj_out
        loss = ((out - target) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            proj_in -= lr * proj_in.grad
            proj_out -= lr * proj_out.grad
            proj_in.grad.zero_()
            proj_out.grad.zero_()

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
