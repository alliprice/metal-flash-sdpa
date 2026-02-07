# metal-flash-sdpa

Drop-in acceleration for `F.scaled_dot_product_attention` on Apple Silicon MPS devices. Monkey-patches PyTorch's SDPA to dispatch eligible calls to Metal Flash Attention v2 kernels (from [ccv](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)). Forward + backward pass — works for both inference and training.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- PyTorch 2.0+ with MPS support

## Installation

```bash
pip install metal-flash-sdpa
```

Or from source:

```bash
git clone https://github.com/alliprice/metal-flash-sdpa
cd metal-flash-sdpa
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn.functional as F
import metal_flash_sdpa

metal_flash_sdpa.enable()  # monkey-patches F.scaled_dot_product_attention

# Use SDPA as normal — MFA dispatches automatically on MPS
q = torch.randn(1, 8, 2048, 64, device="mps", dtype=torch.float16)
k = torch.randn(1, 8, 2048, 64, device="mps", dtype=torch.float16)
v = torch.randn(1, 8, 2048, 64, device="mps", dtype=torch.float16)

out = F.scaled_dot_product_attention(q, k, v)  # uses Metal Flash Attention
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal masking works too

metal_flash_sdpa.disable()  # restore original SDPA
```

## Training Example

```python
import torch
import metal_flash_sdpa

metal_flash_sdpa.enable()

# Works with any model that uses F.scaled_dot_product_attention internally
model = YourTransformerModel().to("mps")
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    loss = model(batch)
    loss.backward()  # backward pass uses MFA too
    optimizer.step()
    optimizer.zero_grad()
```

## API

| Function | Description |
|---|---|
| `metal_flash_sdpa.enable()` | Monkey-patch `F.scaled_dot_product_attention` to use MFA on MPS |
| `metal_flash_sdpa.disable()` | Restore original SDPA |
| `metal_flash_sdpa.get_dispatch_count()` | Number of times MFA was dispatched since last reset |
| `metal_flash_sdpa.reset_dispatch_count()` | Reset dispatch counter to zero |
| `metal_flash_sdpa.MIN_SEQ_LEN` | Minimum sequence length to dispatch to MFA (default: 256) |

## When MFA Dispatches vs. Falls Back

MFA handles:
- MPS device tensors
- fp16, bf16, fp32
- Sequence length >= 256 (configurable via `MIN_SEQ_LEN`)
- `is_causal=True` or `is_causal=False`
- No attention mask, or all-True boolean masks

Falls back to PyTorch's built-in SDPA for:
- Non-MPS devices
- Dropout (`dropout_p > 0`)
- Non-trivial attention masks (float masks, boolean masks with False values)
- Sequence length < 256 (MFA overhead exceeds benefit)
- `enable_gqa=True` (not yet supported)
- Nested tensors

## Benchmarks

### Microbenchmarks (M3 Pro 36GB)

| Operation | Seq Length | Speedup vs MPS SDPA |
|-----------|-----------|---------------------|
| Forward fp16 | 2048 | 5.88x |
| Forward fp16 | 128 | 2.22x |
| Fwd+Bwd fp16 | 2048 | 2.31x |
| Fwd+Bwd fp16 | 4096 | 3.25x |

### Real-World Training (M3 Pro 36GB)

Qwen-Image 20B LoRA fine-tuning via SimpleTuner (1024px, 20 steps):

| Metric | MFA | Baseline | Speedup |
|--------|-----|----------|---------|
| Per-step (steady state) | ~72s | ~110s | **1.53x** |
| Wall-clock (20 steps) | 1,978s | 2,751s | **1.39x** |

Loss values identical between runs (same seed) — no accuracy impact.

## How It Works

Architecture:

```
F.scaled_dot_product_attention (monkey-patched)
  → dispatch check (device, dtype, seq_len, dropout, mask)
  → MetalFlashAttentionForward (torch.autograd.Function)
    → transpose [B,H,S,D] → [B,S,H,D]
    → C++ bridge (csrc/mfa_bridge.mm)
      → extract MTLBuffer pointers (zero-copy)
      → ccv Metal Flash Attention v2 shader generator
        → runtime-compiled Metal compute shaders
    → transpose output back to [B,H,S,D]
```

The C++ extension extracts raw `MTLBuffer` pointers from PyTorch MPS tensors (zero-copy) and passes them to ccv's Metal Flash Attention v2 runtime shader generator. Shaders are compiled once and cached. Both forward and backward kernels run as Metal compute dispatches on the MPS command queue.

## Kernel Source

The attention kernels come from [ccv's Metal Flash Attention implementation](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) (v2 runtime shader generator, not precompiled metallib). This is the same kernel used by [Draw Things](https://drawthings.ai/) for on-device inference and training.

## Gradient Correctness

Validated with `torch.autograd.gradcheck` and direct comparison against PyTorch's math SDPA:

| Dtype | Forward max diff | Backward max diff |
|-------|-----------------|-------------------|
| fp32 | ~1e-6 | ~1e-6 |
| fp16 | ~5e-4 | ~2e-3 |
| bf16 | ~1e-3 | ~1e-3 |

## License

MIT

## Acknowledgments

- [Metal Flash Attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner — original Swift implementation
- [ccv](https://github.com/liuliu/ccv) by Liu Liu — C++ Metal Flash Attention implementation used here
- [Draw Things](https://engineering.drawthings.ai/) — Metal Flash Attention 2.0 engineering blog post
