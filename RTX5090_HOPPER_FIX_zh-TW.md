# RTX 5090 (Hopper 架構) PyTorch 2.8 相容性修復指南

## 問題摘要

在 RTX 5090 (Hopper 架構) GPU 上使用 PyTorch 2.8.0+cu128 執行 ControlVideo 時遇到兩個主要問題：

1. **xformers Flash Attention 核心錯誤**：Hopper GPU 上的 xformers 0.0.32.post2 會觸發 CUDA invalid argument 錯誤
2. **記憶體溢出 (OOM)**：原始的 attention 實作使用 `torch.baddbmm` 建立巨大的 attention score 張量 (嘗試分配 112.5 GiB)

## 環境資訊

- **GPU**: NVIDIA RTX 5090 (Hopper 架構)
- **CUDA**: 12.8 / 13.0
- **PyTorch**: 2.8.0+cu128
- **xformers**: 0.0.32.post2
- **diffusers**: 0.35.2
- **transformers**: 最新版本

## 解決方案概述

**核心策略**：保留 xformers 套件但繞過其 flash attention 核心，改用 PyTorch 原生的 `F.scaled_dot_product_attention` 搭配 memory-efficient 核心。

### 為什麼不直接停用 xformers？

直接移除或全域停用 xformers 會導致：
- 記憶體使用量大幅增加
- 可能觸發 OOM 錯誤
- 效能下降

## 詳細修改步驟

### 1. 修改 `models/attention.py`

#### 1.1 新增 import

在檔案開頭新增 `nullcontext` 的 import：

```python
from dataclasses import dataclass
from typing import Optional, Callable
import math
import torch
import torch.nn.functional as F
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding2D
from contextlib import nullcontext  # 新增這行
```

#### 1.2 替換 `FullyFrameAttention._attention()` 方法

**原因**：原始實作使用 `torch.baddbmm` 建立完整的 attention score 矩陣，在處理長序列時會消耗大量記憶體。

**位置**：約在第 351 行

**原始程式碼** (需刪除)：
```python
def _attention(self, query, key, value, attention_mask=None):
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    attention_scores = torch.baddbmm(
        torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=self.scale,
    )
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(value.dtype)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    return hidden_states
```

**新的程式碼**：
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

    # Prepare attention mask if provided (expected shape broadcastable to (B,H,N,M))
    attn_mask = None
    if attention_mask is not None:
        # If user passed (B*H, N, M) reshape; else let PyTorch broadcast
        if attention_mask.dim() == 3 and attention_mask.shape[0] == B_times_H:
            attn_mask = attention_mask.view(B, H, N, M)
        else:
            attn_mask = attention_mask

    # Force memory efficient kernel; disable flash on Hopper to avoid invalid argument error
    try:
        cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    except Exception:
        cm = nullcontext()
    with cm:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    # Cast back if we upcasted
    if self.upcast_attention and out.dtype != torch.float16:
        desired_dtype = value.dtype if value.dtype in (torch.float16, torch.bfloat16) else out.dtype
        out = out.to(desired_dtype)

    out = out.reshape(B_times_H, N, D)  # 使用 reshape 而非 view
    out = self.reshape_batch_dim_to_heads(out)
    return out
```

**關鍵改進**：
- 使用 `F.scaled_dot_product_attention` 避免建立巨大的 score 矩陣
- 強制使用 memory-efficient 核心並停用 flash attention（Hopper 上會出錯）
- 使用 `reshape` 而非 `view` 處理非連續張量

### 2. 修改 `models/controlnet_attention.py`

#### 2.1 新增 import

與 `attention.py` 相同，新增：

```python
from contextlib import nullcontext
```

#### 2.2 替換 `IndividualAttention._attention()` 方法

**位置**：約在第 370 行

採用與上述相同的策略，將原始的 `torch.baddbmm` 實作替換為：

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

    out = out.reshape(B_times_H, N, D)  # 使用 reshape 而非 view
    out = self.reshape_batch_dim_to_heads(out)
    return out
```

### 3. 修改 `inference.py`

#### 3.1 停用 xformers 路徑

**位置**：pipeline 建立後，約在第 115-128 行

**原始程式碼** (需刪除或註解)：
```python
# pipe.enable_xformers_memory_efficient_attention()  # 刪除這行
```

**新增程式碼**：
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

**說明**：
- 遍歷 UNet 和 ControlNet 的所有模組
- 將 `_use_memory_efficient_attention_xformers` 標記設為 `False`
- 這會強制使用我們修改過的 `_attention()` 方法而非 xformers 路徑

### 4. 已存在的輔助函數

確認以下函數已在兩個 attention 檔案中定義（應該在之前的修復中已加入）：

```python
def is_xformers_available():
    """Helper function to check xformers availability"""
    return XFORMERS_AVAILABLE
```

## 驗證修復

執行測試命令：

```bash
cd /path/to/ControlVideo
conda activate controlvideo

export CONDITION="depth_midas"
export WIDTH=512
export HEIGHT=512
export VERSION="v10"
export SEED=42
export OUTPUT_BASE="outputs"

python inference.py \
  --prompt "A blue bowl and a yellow bowl are moving on the table." \
  --condition "$CONDITION" \
  --video_path "assets/o2o/3ball.mp4" \
  --output_path "$OUTPUT_BASE/o2o/3ball" \
  --video_length 15 \
  --width $WIDTH \
  --height $HEIGHT \
  --version $VERSION \
  --seed $SEED
```

**預期結果**：
- ✅ 順利完成 50/50 denoising steps
- ✅ 無 CUDA invalid argument 錯誤
- ✅ 無 OOM 錯誤
- ✅ 輸出影片保存至 `outputs/o2o/3ball/`
- ⚠️  可能出現 `torch.backends.cuda.sdp_kernel()` 的 FutureWarning（可忽略或升級）

## 技術原理

### 為什麼 scaled_dot_product_attention 更好？

1. **融合操作**：PyTorch 的 SDPA 是單一融合操作，不會建立中間的巨大 score 矩陣
2. **記憶體效率**：使用 memory-efficient attention 演算法（類似 xformers 但相容 Hopper）
3. **核心選擇**：
   - Flash Attention：快但在 Hopper 上與某些 xformers 版本不相容
   - Memory-efficient：稍慢但穩定且記憶體友善
   - Math fallback：最慢但保證可用

### 為什麼使用 reshape 而非 view？

`F.scaled_dot_product_attention` 的輸出可能不是連續的記憶體佈局。使用 `reshape` 會在必要時建立副本，避免 `RuntimeError: view size is not compatible with input tensor's size and stride`。

## 效能考量

- **記憶體使用**：大幅降低（從嘗試分配 112.5 GiB 降至合理範圍）
- **速度**：Memory-efficient 核心略慢於 Flash Attention，但在 Hopper 上是唯一穩定選項
- **品質**：輸出品質完全相同（數學上等價的操作）

## 未來改進（選用）

### 1. 移除棄用警告

將 `torch.backends.cuda.sdp_kernel` 替換為 `torch.nn.attention.sdpa_kernel`（PyTorch 2.8+ 推薦）：

```python
# 舊版（會有 FutureWarning）
cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

# 新版（推薦）
from torch.nn.attention import SDPBackend, sdpa_kernel
cm = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
```

### 2. 新增環境變數切換

允許在非 Hopper GPU 上重新啟用 xformers：

```python
import os

USE_XFORMERS = os.getenv("CONTROLVIDEO_USE_XFORMERS", "0") == "1"
if not USE_XFORMERS:
    # 停用 xformers 的現有程式碼
    ...
```

### 3. GPU 架構自動偵測

```python
gpu_arch = torch.cuda.get_device_capability()
is_hopper = gpu_arch[0] >= 9  # Hopper 是 compute capability 9.0+
if is_hopper:
    # 強制使用 SDPA
    ...
```

## 適用於其他專案

此修復策略也適用於：
- **Ground-A-Video**：使用相同的 attention 模式
- **AnimateDiff**：類似的 temporal attention
- **任何基於 diffusers 的專案**：在 Hopper GPU 上遇到 xformers flash attention 問題

### 通用修復模板

1. 找到專案中的 attention 實作（通常在 `models/attention*.py`）
2. 將 `torch.baddbmm` + `softmax` + `torch.bmm` 替換為 `F.scaled_dot_product_attention`
3. 在 pipeline 建立後停用 xformers 標記
4. 確保使用 `reshape` 而非 `view` 處理輸出張量

## 參考資源

- [PyTorch Scaled Dot Product Attention 文件](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch Memory-Efficient Attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel)
- [xformers GitHub Issues](https://github.com/facebookresearch/xformers/issues)

## 疑難排解

### 問題：仍然出現 OOM
**解決方案**：
- 降低 `video_length`
- 降低 `width` 和 `height`（必須是 32 的倍數）
- 啟用 gradient checkpointing（如果模型支援）

### 問題：執行速度很慢
**解決方案**：
- 檢查是否成功停用 xformers 路徑
- 考慮降低 `num_inference_steps`（從 50 降至 30-40）
- 確認 GPU 未被其他程序佔用

### 問題：輸出品質下降
**解決方案**：
- 此修復應該不會影響品質（數學上等價）
- 如果確實有差異，檢查 dtype 轉換（應保持 fp16）
- 調整 `guidance_scale` 參數

## 總結

此修復透過以下方式解決 RTX 5090 相容性問題：
1. ✅ 避免 xformers flash attention 在 Hopper 上的 CUDA 錯誤
2. ✅ 使用 PyTorch 原生 SDPA 防止記憶體溢出
3. ✅ 保持與舊版 GPU 的相容性
4. ✅ 無需移除 xformers 套件（避免其他潛在問題）

**修改檔案清單**：
- `models/attention.py`：新增 nullcontext import，替換 `_attention()` 方法
- `models/controlnet_attention.py`：新增 nullcontext import，替換 `_attention()` 方法
- `inference.py`：停用 xformers 標記，強制使用 SDPA 路徑

---

**文件版本**：1.0  
**最後更新**：2025-11-08  
**測試環境**：RTX 5090, PyTorch 2.8.0+cu128, xformers 0.0.32.post2  
**作者**：根據 ControlVideo + Ground-A-Video 修復經驗編寫
