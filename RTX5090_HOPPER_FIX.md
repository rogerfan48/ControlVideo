# RTX 5090 (Hopper) + PyTorch 2.8 Compatibility Fix Guide

## Problem Summary
Running ControlVideo on an RTX 5090 (Hopper) with PyTorch 2.8.0+cu128 hit two major issues:

1. xformers flash attention CUDA error (invalid argument) on Hopper with xformers 0.0.32.post2
2. Out-of-memory (OOM) from the naive attention implementation that materializes a huge attention score tensor via `torch.baddbmm` (attempted to allocate 112.5 GiB)

## Environment
- GPU: NVIDIA RTX 5090 (Hopper)
- CUDA: 12.8 / 13.0
- PyTorch: 2.8.0+cu128
- xformers: 0.0.32.post2
- diffusers: 0.35.2
- transformers: latest

## Solution Overview
Core strategy: Keep xformers installed but bypass its flash attention kernels on Hopper. Use PyTorch’s native `F.scaled_dot_product_attention` (SDPA) with the memory‑efficient backend.

Why not disable/uninstall xformers globally?
- Doing so often increases memory usage and may cause OOM
- Potential performance loss elsewhere

## Detailed Changes

### 1) File: `models/attention.py`

Add nullcontext import near the top:
```python
from contextlib import nullcontext
```

Replace `FullyFrameAttention._attention()` to use SDPA instead of `baddbmm`:
```python
def _attention(self, query, key, value, attention_mask=None):
    """Memory-friendly attention using PyTorch scaled_dot_product_attention to avoid giant score tensors.

    Expects query/key/value of shape (B*H, N, D). We reshape to (B,H,N,D) for the fused op.
    """
    B_times_H, N, D = query.shape
    H = self.heads
    B = B_times_H // H
    _, M, _ = key.shape

    if self.upcast_attention:
        query = query.float(); key = key.float(); value = value.float()

    query = query.view(B, H, N, D)
    key = key.view(B, H, M, D)
    value = value.view(B, H, M, D)

    attn_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 3 and attention_mask.shape[0] == B_times_H:
            attn_mask = attention_mask.view(B, H, N, M)
        else:
            attn_mask = attention_mask

    try:
        cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    except Exception:
        cm = nullcontext()
    with cm:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    if self.upcast_attention and out.dtype != torch.float16:
        desired_dtype = value.dtype if value.dtype in (torch.float16, torch.bfloat16) else out.dtype
        out = out.to(desired_dtype)

    out = out.reshape(B_times_H, N, D)  # use reshape, not view
    out = self.reshape_batch_dim_to_heads(out)
    return out
```

Key improvements:
- Avoids explicit (B*H, N, M) score tensor materialization
- Forces memory‑efficient kernel and disables flash kernel on Hopper
- Uses reshape (not view) for possibly non-contiguous outputs

### 2) File: `models/controlnet_attention.py`

Add nullcontext import:
```python
from contextlib import nullcontext
```

Replace `IndividualAttention._attention()` similarly:
```python
def _attention(self, query, key, value, attention_mask=None):
    """Memory-friendly attention using PyTorch scaled_dot_product_attention.

    query/key/value: (B*H, N, D)
    """
    B_times_H, N, D = query.shape
    H = self.heads
    B = B_times_H // H
    _, M, _ = key.shape

    if self.upcast_attention:
        query = query.float(); key = key.float(); value = value.float()

    query = query.view(B, H, N, D)
    key = key.view(B, H, M, D)
    value = value.view(B, H, M, D)

    attn_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 3 and attention_mask.shape[0] == B_times_H:
            attn_mask = attention_mask.view(B, H, N, M)
        else:
            attn_mask = attention_mask

    try:
        cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    except Exception:
        cm = nullcontext()
    with cm:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    if self.upcast_attention and out.dtype != torch.float16:
        desired_dtype = value.dtype if value.dtype in (torch.float16, torch.bfloat16) else out.dtype
        out = out.to(desired_dtype)

    out = out.reshape(B_times_H, N, D)  # reshape not view
    out = self.reshape_batch_dim_to_heads(out)
    return out
```

### 3) File: `inference.py`

Ensure xformers fast path is disabled after pipeline creation (while keeping xformers installed):
```python
pipe.enable_vae_slicing()
# Disable xformers flash kernel on Hopper; rely on PyTorch SDPA fallback already patched in attention modules
try:
    if hasattr(pipe, "unet"):
        for m in pipe.unet.modules():
            if hasattr(m, "_use_memory_efficient_attention_xformers"):
                m._use_memory_efficient_attention_xformers = False
    if hasattr(pipe, "controlnet"):
        for m in pipe.controlnet.modules():
            if hasattr(m, "_use_memory_efficient_attention_xformers"):
                m._use_memory_efficient_attention_xformers = False
except Exception:
    pass
pipe.to(device)
```

This forces the SDPA path we patched above and avoids the Hopper xformers flash kernel.

## Validation
- 50/50 denoising steps complete successfully
- No CUDA invalid argument error
- No OOM
- Outputs saved under `outputs/o2o/3ball/`
- You may see a FutureWarning about `torch.backends.cuda.sdp_kernel()`; safe to ignore or update to the new API (see below)

## Why SDPA?
- Fused op that doesn’t explicitly allocate the NxM score matrix
- Memory‑efficient backend similar to xformers but stable on Hopper
- Backend control lets us disable flash and select efficient kernels

## Performance
- Memory usage: dramatically reduced (no 112.5 GiB allocation)
- Speed: slightly slower than flash attention, but stable and memory‑friendly on Hopper
- Quality: no change (mathematically equivalent attention)

## Optional Improvements

1) Replace deprecated context manager
```python
# Old (emits FutureWarning)
cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

# New (recommended)
from torch.nn.attention import SDPBackend, sdpa_kernel
cm = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
```

2) Environment toggle to re‑enable xformers on non‑Hopper GPUs
```python
import os
USE_XFORMERS = os.getenv("CONTROLVIDEO_USE_XFORMERS", "0") == "1"
if not USE_XFORMERS:
    # keep flags False as above
    ...
```

3) Auto‑detect Hopper
```python
gcu = torch.cuda.get_device_capability()
is_hopper = gcu[0] >= 9  # Hopper is 9.0+
if is_hopper:
    # force SDPA path
    ...
```

## Applies To
- Ground-A-Video (same attention patterns)
- AnimateDiff and other video attention models
- Any diffusers‑based pipeline on Hopper encountering xformers flash attention issues

## File Change Checklist
- models/attention.py: add nullcontext import; replace `_attention()` (use SDPA); use reshape
- models/controlnet_attention.py: add nullcontext import; replace `_attention()` (use SDPA); use reshape
- inference.py: disable `_use_memory_efficient_attention_xformers` flags for UNet and ControlNet

---
Doc version: 1.0  
Last updated: 2025‑11‑08  
Tested on: RTX 5090, PyTorch 2.8.0+cu128, xformers 0.0.32.post2  
Author: Based on fixes adapted from ControlVideo and Ground‑A‑Video
