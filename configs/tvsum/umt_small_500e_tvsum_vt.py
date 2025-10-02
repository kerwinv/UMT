_base_ = [
    '../_base_/models/umt_small.py', '../_base_/plugins/hd.py',
    '../_base_/datasets/tvsum.py', '../_base_/schedules/500e.py',
    '../_base_/runtime.py'
]
# model settings - enable cross-modal gate
# model = dict(
#     cross_enc=dict(
#         gate_cfg=dict(
#             type='CrossModalGate',
#             mode='channel_token',
#             init_bias=-1.0,
#             min_gate=0.0,
#             dropout=0.1,
#             learnable_temp=True,
#             reg_l1=1e-4
#         )
#     )
# )
# dataset settings
data = dict(train=dict(domain='VT'), val=dict(domain='VT'))
