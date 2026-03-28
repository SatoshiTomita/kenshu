from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except Exception:
        return False
    return isinstance(x, torch.Tensor)

# 正規化を行う関数
def minmax_normalize(x: Any, vmin: Any, vmax: Any, eps: float = 1.0e-8):
    if _is_torch_tensor(x) or _is_torch_tensor(vmin) or _is_torch_tensor(vmax):
        import torch

        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        vmin_t = vmin if isinstance(vmin, torch.Tensor) else torch.as_tensor(vmin)
        vmax_t = vmax if isinstance(vmax, torch.Tensor) else torch.as_tensor(vmax)
        denom = torch.maximum(vmax_t - vmin_t, torch.tensor(eps, dtype=x_t.dtype, device=x_t.device))
        return (x_t - vmin_t) / denom

    x_np = np.asarray(x, dtype=np.float32)
    vmin_np = np.asarray(vmin, dtype=np.float32)
    vmax_np = np.asarray(vmax, dtype=np.float32)
    denom = np.maximum(vmax_np - vmin_np, eps)
    return (x_np - vmin_np) / denom

# 逆正規化を行う関数
def minmax_denormalize(x: Any, vmin: Any, vmax: Any, eps: float = 1.0e-8):
    if _is_torch_tensor(x) or _is_torch_tensor(vmin) or _is_torch_tensor(vmax):
        import torch

        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        vmin_t = vmin if isinstance(vmin, torch.Tensor) else torch.as_tensor(vmin)
        vmax_t = vmax if isinstance(vmax, torch.Tensor) else torch.as_tensor(vmax)
        scale = torch.maximum(vmax_t - vmin_t, torch.tensor(eps, dtype=x_t.dtype, device=x_t.device))
        return x_t * scale + vmin_t

    x_np = np.asarray(x, dtype=np.float32)
    vmin_np = np.asarray(vmin, dtype=np.float32)
    vmax_np = np.asarray(vmax, dtype=np.float32)
    scale = np.maximum(vmax_np - vmin_np, eps)
    return x_np * scale + vmin_np
