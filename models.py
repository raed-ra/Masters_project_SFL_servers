"""
Split-model definitions for poisoning-resilient SFL.

The full model is a simple CNN (inspired by VGG-style blocks) that is
**split at a cut layer**:

  ClientModel  – layers 0 … (cut_layer - 1)   → runs on the *client*
  ServerModel  – layers cut_layer … end        → runs on the *SL server*

Both accept a `cut_layer` argument so that you can vary the split point
without changing the architecture.

Architecture summary (4 convolutional blocks + a classifier):
  Block 0: Conv(3→32)  + BN + ReLU + MaxPool
  Block 1: Conv(32→64) + BN + ReLU + MaxPool
  Block 2: Conv(64→128)+ BN + ReLU + MaxPool
  Block 3: Conv(128→256)+ BN + ReLU + AdaptiveAvgPool
  Classifier: FC(256→num_classes)
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------- #
# Building blocks                                                          #
# ---------------------------------------------------------------------- #

def _conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------- #
# Full model (useful for baseline / debugging)                            #
# ---------------------------------------------------------------------- #

class FullCNN(nn.Module):
    """End-to-end CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.block0 = _conv_block(3,   32,  pool=True)   # 32×32 → 16×16
        self.block1 = _conv_block(32,  64,  pool=True)   # 16×16 →  8×8
        self.block2 = _conv_block(64,  128, pool=True)   # 8×8   →  4×4
        self.block3 = _conv_block(128, 256, pool=False)  # 4×4   →  4×4
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------- #
# Client-side model (front portion)                                        #
# ---------------------------------------------------------------------- #

class ClientModel(nn.Module):
    """
    Front portion of the split CNN running on the client.

    Parameters
    ----------
    cut_layer   : number of conv blocks to keep on the client.
                  Must be in {1, 2, 3, 4}.
    num_classes : only used when cut_layer == 4 (full model on client).
    """

    _CHANNELS = [(3, 32), (32, 64), (64, 128), (128, 256)]
    _POOL      = [True,    True,     True,      False   ]

    def __init__(self, cut_layer: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        if not 1 <= cut_layer <= 4:
            raise ValueError("cut_layer must be between 1 and 4 (inclusive)")
        self.cut_layer = cut_layer
        blocks = []
        for i in range(cut_layer):
            in_ch, out_ch = self._CHANNELS[i]
            blocks.append(_conv_block(in_ch, out_ch, pool=self._POOL[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------- #
# Server-side model (back portion)                                         #
# ---------------------------------------------------------------------- #

class ServerModel(nn.Module):
    """
    Back portion of the split CNN running on the SL server.

    Parameters
    ----------
    cut_layer   : must match the ClientModel value.
    num_classes : number of output classes.
    """

    _CHANNELS = [(3, 32), (32, 64), (64, 128), (128, 256)]
    _POOL      = [True,    True,     True,      False   ]

    def __init__(self, cut_layer: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        if not 1 <= cut_layer <= 4:
            raise ValueError("cut_layer must be between 1 and 4 (inclusive)")
        self.cut_layer = cut_layer
        blocks = []
        for i in range(cut_layer, 4):
            in_ch, out_ch = self._CHANNELS[i]
            blocks.append(_conv_block(in_ch, out_ch, pool=self._POOL[i]))
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        x = self.blocks(smashed_data)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------- #
# Utility: count parameters                                                #
# ---------------------------------------------------------------------- #

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
