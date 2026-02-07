"""Forward pass correctness tests for Metal Flash Attention."""
import pytest
import torch
import metal_flash_sdpa


def reference_sdpa(Q, K, V, scale):
    """Eager attention on CPU as reference."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


FORWARD_CONFIGS = [
    # (B, H, R, C, D, dtype, Hk)
    (1, 1, 64, 64, 64, torch.float16, None),
    (2, 8, 128, 128, 64, torch.float16, None),
    (1, 4, 256, 256, 128, torch.float16, None),
    (4, 8, 512, 512, 64, torch.float16, None),
    (2, 4, 128, 128, 64, torch.float32, None),
    (1, 4, 64, 256, 64, torch.float16, None),
    (1, 4, 256, 64, 64, torch.float16, None),
    (1, 8, 2048, 2048, 64, torch.float16, None),
    (1, 4, 256, 256, 64, torch.bfloat16, None),
    (2, 8, 512, 512, 64, torch.bfloat16, None),
]


def _config_id(config):
    B, H, R, C, D, dtype, Hk = config
    dtype_name = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
    gqa = f"-Hk{Hk}" if Hk is not None else ""
    return f"{dtype_name}-B{B}-H{H}-{R}x{C}-D{D}{gqa}"


@pytest.mark.parametrize("B,H,R,C,D,dtype,Hk", FORWARD_CONFIGS, ids=[_config_id(c) for c in FORWARD_CONFIGS])
def test_forward_correctness(B, H, R, C, D, dtype, Hk):
    if Hk is None:
        Hk = H

    Q = torch.randn(B, H, R, D, device="mps", dtype=dtype)
    K = torch.randn(B, Hk, C, D, device="mps", dtype=dtype)
    V = torch.randn(B, Hk, C, D, device="mps", dtype=dtype)

    scale = D ** -0.5

    metal_flash_sdpa.enable()
    with torch.no_grad():
        out_mfa = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

    out_ref = reference_sdpa(
        Q.cpu().float(), K.cpu().float(), V.cpu().float(), scale
    ).to(dtype).to("mps")

    max_diff = (out_mfa - out_ref).abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 0.05
    assert max_diff < tol, f"Output mismatch: max_diff={max_diff} > tol={tol}"
