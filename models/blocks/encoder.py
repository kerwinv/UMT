# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
from nncore.nn import MODELS, build_linear_modules, build_model, build_norm_layer


@MODELS.register()
class UniModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 p=0.5,
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(UniModalEncoder, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = build_norm_layer(drop_cfg)
        self.mapping = build_linear_modules(dims, **kwargs)
        self.pos_enc = build_model(pos_cfg, enc_dims)
        self.encoder = build_model(enc_cfg, enc_dims, bundler='sequential')
        self.norm = build_norm_layer(norm_cfg, enc_dims)

    def forward(self, x, **kwargs):
        if self.dropout is not None:
            x = self.dropout(x)
        if self.mapping is not None:
            x = self.mapping(x)
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(x)
            x = self.encoder(x, pe=pe, **kwargs)
        if self.norm is not None:
            x = self.norm(x)
        return x


@MODELS.register()
class CrossModalGate(nn.Module):
    """
    跨模态门控:
      mode=scalar: 全局标量
      mode=channel: 通道向量
      mode=channel_token: 通道 + 样本级标量 (轻量 token 维度广播)
    """
    def __init__(self, dims, mode='scalar', init_bias=-2.0, min_gate=0.0, dropout=0.0):
        super().__init__()
        self.mode = mode
        self.min_gate = float(min_gate)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == 'scalar':
            self.param = nn.Parameter(torch.full((1,), init_bias))
        elif mode == 'channel':
            self.param = nn.Parameter(torch.full((dims,), init_bias))
        elif mode == 'channel_token':
            self.param = nn.Parameter(torch.full((dims,), init_bias))  # 通道门
            self.q_proj = nn.Linear(dims, dims, bias=False)
            self.s_proj = nn.Linear(dims, dims, bias=False)
            self.mix_proj = nn.Linear(dims, 1, bias=True)  # 生成样本级标量
        else:
            raise ValueError(f'Unsupported gate mode: {mode}')

    def _activate(self, raw):
        g = torch.sigmoid(raw)
        if self.min_gate > 0:
            g = self.min_gate + (1 - self.min_gate) * g
        return g

    def forward(self, target, source):
        """
        target: 主模态 (B, Lt, D)
        source: 被调节模态 (B, Ls, D)
        """
        if self.mode == 'scalar':
            g = self._activate(self.param).view(1, 1, 1)          # (1,1,1)
            return self.dropout(source * g)
        if self.mode == 'channel':
            g = self._activate(self.param).view(1, 1, -1)         # (1,1,D)
            return self.dropout(source * g)
        # channel_token
        B, Ls, D = source.shape
        tgt_pool = target.mean(dim=1)    # (B,D)
        src_pool = source.mean(dim=1)    # (B,D)
        joint = torch.tanh(self.q_proj(tgt_pool) + self.s_proj(src_pool))
        token_scale = self._activate(self.mix_proj(joint)).view(B, 1, 1)   # (B,1,1)
        ch_scale = self._activate(self.param).view(1, 1, D)                # (1,1,D)
        return self.dropout(source * token_scale * ch_scale)


@MODELS.register()
class CrossModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 fusion_type='sum',
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 gate_cfg=None,
                 **kwargs):
        super(CrossModalEncoder, self).__init__()
        assert fusion_type in ('sum', 'mean', 'concat')

        map_dims = [2 * dims, dims] if fusion_type == 'concat' else None
        self.fusion_type = fusion_type

        self.pos_enc = build_model(pos_cfg, dims)
        self.encoder = build_model(enc_cfg, dims)
        self.mapping = build_linear_modules(map_dims, **kwargs)
        self.norm = build_norm_layer(norm_cfg, dims)

        self.gate = None
        if gate_cfg is not None:
            assert gate_cfg.get('type') == 'CrossModalGate'
            self.gate = MODELS.build(dict(
                type='CrossModalGate',
                dims=dims,
                mode=gate_cfg.get('mode', 'scalar'),
                init_bias=gate_cfg.get('init_bias', -2.0),
                min_gate=gate_cfg.get('min_gate', 0.0),
                dropout=gate_cfg.get('dropout', 0.0),
            ))

    def forward(self, x_v, x_a, **kwargs):
        """
        x_v: (B, Lv, D) 主模态
        x_a: (B, La, D) 次模态（被调节）
        """
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(x_v)
            x_v, x_a = self.encoder(x_v, x_a, pe=pe, **kwargs)
        if self.gate is not None:
            x_a = self.gate(x_v, x_a)
        if self.fusion_type in ('sum', 'mean'):
            x = (x_v + x_a) / ((self.fusion_type == 'mean') + 1)
        else:
            x = torch.cat((x_v, x_a), dim=-1)
            x = self.mapping(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
