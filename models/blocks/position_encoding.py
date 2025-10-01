# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import math
import torch
import torch.nn as nn
from nncore.nn import MODELS


def _factor_hw(seq_len: int):
    """寻找接近正方形的 (h, w)，找不到合适分解则抛错（避免 1×L 降级导致语义偏差）"""
    h = int(math.sqrt(seq_len))
    best = None
    for i in range(h, 0, -1):
        if seq_len % i == 0:
            best = (i, seq_len // i)
            break
    if best is None:
        raise ValueError(f"Cannot factor seq_len={seq_len} into (h, w)")
    return best  # (h, w)


@MODELS.register()
class Rotary2DPositionalEncoding(nn.Module):
    """
    2D 旋转（RoPE 风格）位置编码（直接对输入特征做旋转替换，而非只用于 Q/K）。
    单模态场景：假设序列可以映射到规则 2D 网格 (height, width)。

    Args:
        dims (int): 输入特征维度，需被4整除（x,y方向各一半，且需成对偶数）。
        max_len (int): 预留（当前未用，可扩展缓存策略）。
        height (int|None): 网格高度（建议显式传入）。
        width (int|None): 网格宽度。
        base (float): 频率基数。
        learned (bool): 是否学习频率（对 inv_freq 添加可学习扰动）。
        cache (bool): 是否按 (seq_len, device, dtype) 缓存 cos/sin。
    """

    def __init__(self, dims, max_len=5000, height=None, width=None,
                 base=10000.0, learned=False, cache=True):
        super().__init__()
        assert dims % 4 == 0, f"dims {dims} must be divisible by 4"
        self.dims = dims
        self.max_len = max_len
        self.height = height
        self.width = width
        self.base = base
        self.learned = learned
        self.cache = cache

        self.d_x = dims // 2
        self.d_y = dims // 2

        # 频率（inv_freq），长度应为各方向一半维度：d_x/2, d_y/2
        inv_freq_x = self._get_frequencies(self.d_x)  # shape: (d_x/2,)
        inv_freq_y = self._get_frequencies(self.d_y)

        if learned:
            # 可学习扰动，初始化为 0，基频固定更稳定
            self.register_buffer("inv_freq_x_base", inv_freq_x, persistent=True)
            self.register_buffer("inv_freq_y_base", inv_freq_y, persistent=True)
            self.freq_x_delta = nn.Parameter(torch.zeros_like(inv_freq_x))
            self.freq_y_delta = nn.Parameter(torch.zeros_like(inv_freq_y))
        else:
            self.register_buffer("freq_x", inv_freq_x, persistent=True)
            self.register_buffer("freq_y", inv_freq_y, persistent=True)

        # 缓存 (seq_len, device, dtype) -> (x_cos, x_sin, y_cos, y_sin)
        self._angle_cache = {}

    def _get_frequencies(self, dim_direction: int):
        """
        给定某方向子向量维度 dim_direction（例如 d_x），返回长度 dim_direction/2 的 inv_freq。
        之前的错误：传入 dim/2 再步长2，导致得到 dim/4。
        """
        # arange(0, dim_direction, 2) -> 长度 = dim_direction / 2
        return 1.0 / (self.base ** (torch.arange(0, dim_direction, 2, dtype=torch.float32) / dim_direction))

    def _resolve_hw(self, seq_len: int):
        if self.height is not None and self.width is not None:
            assert self.height * self.width == seq_len, \
                f"height*width ({self.height}*{self.width}) != seq_len({seq_len})"
            return self.height, self.width
        h, w = _factor_hw(seq_len)
        if h * w != seq_len:
            raise ValueError(f"Cannot map seq_len={seq_len} to 2D grid cleanly.")
        return h, w

    def _get_positions(self, seq_len: int, device, dtype):
        h, w = self._resolve_hw(seq_len)
        y_pos, x_pos = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij'
        )
        return x_pos.flatten(), y_pos.flatten()  # (seq_len,), (seq_len,)

    def _get_freq_params(self):
        if self.learned:
            return (self.inv_freq_x_base + self.freq_x_delta,
                    self.inv_freq_y_base + self.freq_y_delta)
        return self.freq_x, self.freq_y

    def _get_cos_sin(self, seq_len: int, device, dtype):
        cache_key = (seq_len, device, dtype)
        if self.cache and cache_key in self._angle_cache:
            return self._angle_cache[cache_key]

        x_pos, y_pos = self._get_positions(seq_len, device, dtype)  # (L,)
        freq_x, freq_y = self._get_freq_params()  # (d_x/2,), (d_y/2,)

        # (L, d_x/2)
        x_angles = x_pos.unsqueeze(-1) * freq_x.unsqueeze(0)
        y_angles = y_pos.unsqueeze(-1) * freq_y.unsqueeze(0)

        x_cos, x_sin = torch.cos(x_angles), torch.sin(x_angles)
        y_cos, y_sin = torch.cos(y_angles), torch.sin(y_angles)

        if self.cache:
            self._angle_cache[cache_key] = (x_cos, x_sin, y_cos, y_sin)
        return x_cos, x_sin, y_cos, y_sin

    @staticmethod
    def _apply_rotary_emb(x, cos, sin):
        """
        x: (B, L, D_dir)
        cos/sin: (L, D_dir/2)
        返回同形状旋转后张量
        """
        x1 = x[..., 0::2]  # (B, L, D_dir/2)
        x2 = x[..., 1::2]
        # 利用广播: (B,L,*) * (L,*) -> (B,L,*) 需要在前面对齐
        # cos/sin (L, d/2) -> (1,L,d/2)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
        return out

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, dims)
        return: (B, L, dims)
        """
        B, L, D = x.shape
        assert D == self.dims, f"Input last dim {D} != configured dims {self.dims}"
        device = x.device
        dtype = x.dtype

        x_cos, x_sin, y_cos, y_sin = self._get_cos_sin(L, device, torch.float32)  # cos/sin 用 float32 精度
        # 若输入是 fp16/bf16，不强制转 cos/sin，广播自动提升或需要显式转换
        if x.dtype != torch.float32:
            x_cos = x_cos.to(x.dtype)
            x_sin = x_sin.to(x.dtype)
            y_cos = y_cos.to(x.dtype)
            y_sin = y_sin.to(x.dtype)

        x_feat = x[..., :self.d_x]
        y_feat = x[..., self.d_x:]

        x_rot = self._apply_rotary_emb(x_feat, x_cos, x_sin)
        y_rot = self._apply_rotary_emb(y_feat, y_cos, y_sin)

        return torch.cat([x_rot, y_rot], dim=-1)


@MODELS.register()
class Rotary2DPositionalEncodingAdditive(nn.Module):
    """
    加性 2D 正弦位置编码（非旋转替换版），返回可直接与特征相加的编码。
    """

    def __init__(self, dims, max_len=5000, height=None, width=None, base=10000.0, cache=True):
        super().__init__()
        assert dims % 4 == 0, "dims must be divisible by 4 (x,y 各一半，且需偶数对)"
        self.dims = dims
        self.max_len = max_len
        self.height = height
        self.width = width
        self.base = base
        self.cache = cache

        self.d_x = dims // 2
        self.d_y = dims // 2

        # 两方向 inv_freq
        self.register_buffer(
            "freqs_x",
            1.0 / (base ** (torch.arange(0, self.d_x, 2, dtype=torch.float32) / self.d_x)),
            persistent=True
        )
        self.register_buffer(
            "freqs_y",
            1.0 / (base ** (torch.arange(0, self.d_y, 2, dtype=torch.float32) / self.d_y)),
            persistent=True
        )

        self._cache = {}

    def _resolve_hw(self, seq_len: int):
        if self.height is not None and self.width is not None:
            assert self.height * self.width == seq_len, \
                f"height*width ({self.height}*{self.width}) != seq_len({seq_len})"
            return self.height, self.width
        return _factor_hw(seq_len)

    def _build(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        if self.cache and key in self._cache:
            return self._cache[key]

        h, w = self._resolve_hw(seq_len)
        y_pos, x_pos = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij'
        )
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()

        pe = torch.zeros(seq_len, self.dims, device=device, dtype=dtype)

        # X 部分
        x_angles = x_pos.unsqueeze(-1) * self.freqs_x.unsqueeze(0).to(dtype)
        pe[:, 0:self.d_x:2] = torch.sin(x_angles)
        pe[:, 1:self.d_x:2] = torch.cos(x_angles)

        # Y 部分
        y_angles = y_pos.unsqueeze(-1) * self.freqs_y.unsqueeze(0).to(dtype)
        # 偏移 self.d_x 后再交错填充
        pe[:, self.d_x:self.dims:2] = torch.sin(y_angles)
        pe[:, self.d_x+1:self.dims:2] = torch.cos(y_angles)

        if self.cache:
            self._cache[key] = pe
        return pe

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, dims)
        return: (B, L, dims) 位置编码，可直接加到 x 上
        """
        B, L, D = x.shape
        assert D == self.dims
        pe = self._build(L, x.device, torch.float32)
        if x.dtype != torch.float32:
            pe = pe.to(x.dtype)
        return pe.unsqueeze(0).expand(B, -1, -1)