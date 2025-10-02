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
    def __init__(self, dims, mode='channel_token', init_bias=-1.0,
                 min_gate=0.0, dropout=0.1, learnable_temp=False, reg_l1=0.0):
        super().__init__()
        self.mode = mode
        self.min_gate = float(min_gate)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.learnable_temp = learnable_temp
        self.reg_l1 = reg_l1
        if learnable_temp:
            self.temp = nn.Parameter(torch.tensor(1.0))
        if mode == 'scalar':
            self.param = nn.Parameter(torch.full((1,), init_bias))
        elif mode == 'channel':
            self.param = nn.Parameter(torch.full((dims,), init_bias))
        elif mode == 'channel_token':
            self.param = nn.Parameter(torch.full((dims,), init_bias))
            self.q_proj = nn.Linear(dims, dims, bias=False)
            self.s_proj = nn.Linear(dims, dims, bias=False)
            self.mix_proj = nn.Linear(dims, 1, bias=True)
        else:
            raise ValueError(mode)

    def _activate(self, raw):
        if self.learnable_temp:
            tau = torch.clamp(self.temp, min=0.1)
            raw = raw / tau
        g = torch.sigmoid(raw)
        if self.min_gate > 0:
            g = self.min_gate + (1 - self.min_gate) * g
        return g

    def extra_loss(self):
        if self.reg_l1 <= 0:
            return 0.0
        return self.reg_l1 * self.param.abs().mean()

    def forward(self, target, source):
        if target.dim() != 3 or source.dim() != 3:
            raise ValueError('CrossModalGate expects (B, L, D) tensors.')
        src = self.dropout(source)

        if self.mode == 'scalar':
            gate = self._activate(self.param).view(1, 1, 1)
            gate = gate.expand_as(src)
        elif self.mode == 'channel':
            gate = self._activate(self.param).view(1, 1, -1)
            gate = gate.expand_as(src)
        else:  # channel_token
            channel_gate = self._activate(self.param).view(1, 1, -1)
            channel_gate = channel_gate.expand_as(src)

            q_ctx = target.mean(dim=1)                  # (B, D)
            q_proj = self.q_proj(q_ctx).unsqueeze(1)    # (B, 1, D)
            s_proj = self.s_proj(src)                   # (B, L, D)
            mix = self.mix_proj(torch.tanh(q_proj + s_proj))  # (B, L, 1)
            token_gate = self._activate(mix)
            gate = channel_gate * token_gate

        self._last_gate = gate.detach()
        return src * gate


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
