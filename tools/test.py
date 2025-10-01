import argparse

import torch
try:
    import torch.nn.parallel.scatter_gather as _sg
    if not hasattr(_sg, "_is_namedtuple") and hasattr(_sg, "is_namedtuple"):
        _sg._is_namedtuple = _sg.is_namedtuple
except Exception as e:
    print("namedtuple patch warn:", e)
import nncore

print("torch version:", torch.__version__)
print("torch path:", torch.__file__)

from nncore.engine import Engine, comm
# from nncore.nn import build_model

print("torch version:", torch.__version__)
print("torch path:", torch.__file__)
print("_is_namedtuple in torch.nn.parallel.scatter_gather:", hasattr(torch.nn.parallel.scatter_gather, "_is_namedtuple"))