"""Validate causal masking with real LLM training: small GPT model, A/B vs baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def train_run(use_mfa, n_steps, vocab_size, d_model, n_heads, n_layers,
              max_seq_len, batch_size, seq_len, lr, dtype, seed=42):
    import metal_flash_sdpa

    torch.manual_seed(seed)
    model = MiniGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    model = model.to('mps')
    use_amp = (dtype != torch.float32)
    if not use_amp:
        model = model.to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if use_mfa:
        metal_flash_sdpa.enable()
        metal_flash_sdpa.reset_dispatch_count()
    else:
        metal_flash_sdpa.disable()

    losses = []
    torch.manual_seed(seed + 1000)

    start = time.time()
    for step in range(n_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='mps')
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device='mps')

        # Use autocast for fp16/bf16 to avoid NaN from pure low-precision training
        with torch.autocast('mps', dtype=dtype, enabled=use_amp):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 5 == 0:
            tag = "MFA" if use_mfa else "Baseline"
            print(f"  [{tag}] step {step+1}/{n_steps}: loss={loss.item():.4f}")

    torch.mps.synchronize()
    elapsed = time.time() - start

    dispatch_count = metal_flash_sdpa.get_dispatch_count() if use_mfa else 0
    metal_flash_sdpa.disable()
    return losses, elapsed, dispatch_count


def run_config(vocab_size, d_model, n_heads, n_layers, max_seq_len,
               batch_size, seq_len, n_steps, lr, dtype, dtype_name):
    print(f"\n{'='*60}")
    print(f"  MiniGPT causal training — {dtype_name}")
    print(f"  {n_layers}L, d={d_model}, h={n_heads}, seq={seq_len}, batch={batch_size}")
    print(f"{'='*60}")

    print(f"\nMFA run ({dtype_name}):")
    mfa_losses, mfa_time, dispatches = train_run(
        use_mfa=True, n_steps=n_steps, vocab_size=vocab_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len, batch_size=batch_size, seq_len=seq_len,
        lr=lr, dtype=dtype)

    print(f"\nBaseline run ({dtype_name}):")
    baseline_losses, baseline_time, _ = train_run(
        use_mfa=False, n_steps=n_steps, vocab_size=vocab_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len, batch_size=batch_size, seq_len=seq_len,
        lr=lr, dtype=dtype)

    # Report
    expected_dispatches = n_steps * n_layers
    print(f"\n--- Results ({dtype_name}) ---")
    print(f"MFA dispatches: {dispatches} (expected {expected_dispatches})")
    print(f"MFA time: {mfa_time:.1f}s ({mfa_time/n_steps*1000:.0f}ms/step)")
    print(f"Baseline time: {baseline_time:.1f}s ({baseline_time/n_steps*1000:.0f}ms/step)")
    if baseline_time > 0:
        print(f"Speedup: {baseline_time/mfa_time:.2f}x")

    print(f"\nStep | MFA Loss | Baseline | Diff")
    print(f"-----|----------|----------|--------")
    for i, (m, b) in enumerate(zip(mfa_losses, baseline_losses)):
        print(f"  {i:2d} | {m:8.4f} | {b:8.4f} | {abs(m-b):.6f}")

    max_diff = max(abs(m - b) for m, b in zip(mfa_losses, baseline_losses))
    mfa_decreased = mfa_losses[-1] < mfa_losses[0]
    baseline_decreased = baseline_losses[-1] < baseline_losses[0]
    has_nan = any(x != x for x in mfa_losses)

    print(f"\nMax loss diff: {max_diff:.6f}")
    print(f"MFA loss decreased: {mfa_losses[0]:.4f} -> {mfa_losses[-1]:.4f} {'✓' if mfa_decreased else '✗'}")
    print(f"Baseline decreased: {baseline_losses[0]:.4f} -> {baseline_losses[-1]:.4f} {'✓' if baseline_decreased else '✗'}")
    print(f"NaN in MFA losses: {'YES ✗' if has_nan else 'No ✓'}")

    # Validation
    if has_nan:
        print(f"\n*** FAIL: NaN detected in MFA losses! ***")
        return False
    elif not mfa_decreased:
        print(f"\n*** WARN: MFA loss did not decrease ***")
        return False
    elif dtype == torch.float32 and max_diff > 0.01:
        print(f"\n*** WARN: fp32 loss divergence > 0.01 ***")
        return False
    else:
        print(f"\n*** PASS ***")
        return True


if __name__ == '__main__':
    n_steps = 20
    lr = 1e-3

    results = []

    # Test 1: fp32, seq=256 (correctness — expect exact match)
    results.append(run_config(
        vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
        max_seq_len=512, batch_size=4, seq_len=256,
        n_steps=n_steps, lr=lr, dtype=torch.float32, dtype_name="float32 seq=256"))

    # Test 2: fp16 with autocast, seq=256 (stability)
    results.append(run_config(
        vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
        max_seq_len=512, batch_size=4, seq_len=256,
        n_steps=n_steps, lr=lr, dtype=torch.float16, dtype_name="float16 seq=256 (autocast)"))

    # Test 3: fp16 with autocast, seq=512 (speedup territory)
    results.append(run_config(
        vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
        max_seq_len=1024, batch_size=2, seq_len=512,
        n_steps=n_steps, lr=lr, dtype=torch.float16, dtype_name="float16 seq=512 (autocast)"))

    # Test 4: bf16 with autocast, seq=512
    results.append(run_config(
        vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
        max_seq_len=1024, batch_size=2, seq_len=512,
        n_steps=n_steps, lr=lr, dtype=torch.bfloat16, dtype_name="bfloat16 seq=512 (autocast)"))

    print(f"\n{'='*60}")
    print(f"  Summary: {sum(results)}/{len(results)} passed")
    print(f"{'='*60}")
