"""Integration test: verify metal-flash-sdpa intercepts attention in diffusers/SimpleTuner pipeline."""
import torch
import torch.nn.functional as F


def test_diffusers_dispatch_intercept():
    """Verify that metal_flash_sdpa.enable() intercepts diffusers' native attention backend."""
    import metal_flash_sdpa
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 8, 512, 64
    # diffusers native backend expects [B, seq, heads, head_dim] and permutes internally
    query = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)
    key = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)
    value = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)

    # Without metal-flash-sdpa
    metal_flash_sdpa.disable()
    out_default = dispatch_attention_fn(query, key, value)

    # With metal-flash-sdpa
    metal_flash_sdpa.enable()
    from unittest.mock import patch
    with patch('metal_flash_sdpa.mfa_attention_forward',
               wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        out_mfa = dispatch_attention_fn(query, key, value)
        assert mock_fwd.called, "MFA forward was NOT called through diffusers dispatch!"

    metal_flash_sdpa.disable()

    # Compare outputs
    max_diff = (out_mfa - out_default).abs().max().item()
    print(f"  [PASS] diffusers dispatch intercept: max_diff={max_diff:.6f}, MFA called=True")
    assert max_diff < 0.01, f"Output mismatch: {max_diff}"


def test_diffusers_attention_processor():
    """Test with diffusers Attention module (as used by Qwen-Image transformer blocks)."""
    import metal_flash_sdpa
    from diffusers.models.attention_processor import Attention

    D = 64
    H = 8

    # Create a diffusers Attention module similar to what Qwen-Image uses
    attn = Attention(
        query_dim=D * H,
        heads=H,
        dim_head=D,
    ).to(device='mps', dtype=torch.float16)

    B, R = 1, 512
    hidden_states = torch.randn(B, R, D * H, device='mps', dtype=torch.float16)

    # Without MFA
    metal_flash_sdpa.disable()
    with torch.no_grad():
        out_default = attn(hidden_states)

    # With MFA
    metal_flash_sdpa.enable()
    from unittest.mock import patch
    with patch('metal_flash_sdpa.mfa_attention_forward',
               wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        with torch.no_grad():
            out_mfa = attn(hidden_states)
        mfa_called = mock_fwd.called

    metal_flash_sdpa.disable()

    max_diff = (out_mfa - out_default).abs().max().item()
    print(f"  [PASS] Attention processor: max_diff={max_diff:.6f}, MFA called={mfa_called}")
    assert max_diff < 0.05, f"Output mismatch: {max_diff}"


def test_backward_through_diffusers():
    """Verify gradients flow through diffusers dispatch with MFA enabled."""
    import metal_flash_sdpa
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 4, 512, 64
    query = torch.randn(B, R, H, D, device='mps', dtype=torch.float16, requires_grad=True)
    key = torch.randn(B, R, H, D, device='mps', dtype=torch.float16, requires_grad=True)
    value = torch.randn(B, R, H, D, device='mps', dtype=torch.float16, requires_grad=True)

    metal_flash_sdpa.enable()
    out = dispatch_attention_fn(query, key, value)
    loss = out.sum()
    loss.backward()
    metal_flash_sdpa.disable()

    assert query.grad is not None, "query.grad is None"
    assert key.grad is not None, "key.grad is None"
    assert value.grad is not None, "value.grad is None"
    assert query.grad.abs().sum() > 0, "query.grad is all zeros"
    print("  [PASS] Backward through diffusers dispatch: gradients flow correctly")


def test_small_seq_fallback_in_diffusers():
    """Verify small sequences fall back to original SDPA even through diffusers."""
    import metal_flash_sdpa
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    B, H, R, D = 1, 4, 64, 64  # R=64 < MIN_SEQ_LEN=256
    query = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)
    key = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)
    value = torch.randn(B, R, H, D, device='mps', dtype=torch.float16)

    metal_flash_sdpa.enable()
    from unittest.mock import patch
    with patch('metal_flash_sdpa.mfa_attention_forward') as mock_fwd:
        with torch.no_grad():
            out = dispatch_attention_fn(query, key, value)
        assert not mock_fwd.called, "MFA should NOT be called for small sequences"
    metal_flash_sdpa.disable()
    print(f"  [PASS] Small seq (R={R}) correctly falls back through diffusers")


def test_mfa_call_counter():
    """Count how many times MFA is called during a simulated training step."""
    import metal_flash_sdpa

    call_count = 0
    original_fwd = metal_flash_sdpa.mfa_attention_forward

    def counting_fwd(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_fwd(*args, **kwargs)

    metal_flash_sdpa.enable()
    # Temporarily replace the forward function with our counting wrapper
    import metal_flash_sdpa._C
    orig = metal_flash_sdpa._C.mfa_attention_forward
    metal_flash_sdpa._C.mfa_attention_forward = counting_fwd
    # Also patch the module-level reference
    metal_flash_sdpa.mfa_attention_forward = counting_fwd

    B, H, R, D = 1, 8, 1024, 64
    Q = torch.randn(B, H, R, D, device='mps', dtype=torch.float16, requires_grad=True)
    K = torch.randn(B, H, R, D, device='mps', dtype=torch.float16, requires_grad=True)
    V = torch.randn(B, H, R, D, device='mps', dtype=torch.float16, requires_grad=True)

    out = F.scaled_dot_product_attention(Q, K, V)
    loss = out.sum()
    loss.backward()

    # Restore
    metal_flash_sdpa._C.mfa_attention_forward = orig
    metal_flash_sdpa.mfa_attention_forward = orig
    metal_flash_sdpa.disable()

    print(f"  [PASS] MFA call counter: {call_count} forward calls (1 expected for fwd)")
    assert call_count >= 1, f"Expected at least 1 MFA call, got {call_count}"


if __name__ == '__main__':
    print("=== SimpleTuner Integration Tests ===\n")

    print("Diffusers dispatch intercept:")
    test_diffusers_dispatch_intercept()

    print("\nDiffusers Attention processor:")
    test_diffusers_attention_processor()

    print("\nBackward through diffusers:")
    test_backward_through_diffusers()

    print("\nSmall sequence fallback in diffusers:")
    test_small_seq_fallback_in_diffusers()

    print("\nMFA call counter:")
    test_mfa_call_counter()

    print("\n=== All integration tests passed! ===")
