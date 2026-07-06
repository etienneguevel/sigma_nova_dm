import itertools

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class ConvLayer(nn.Module):

    def __init__(
        self, 
        kernel_size: int, 
        in_channels: int, 
        out_channels: int, 
        max_pool_kernel_size: int
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(max_pool_kernel_size),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class EEGSimpleConv(nn.Module):
    def __init__(
        self,
        fm: int,
        n_convs: int, 
        kernel_size: int,
        n_chan: int,
        n_classes: int,
        n_subjects: int | None = None
    ):
        super(EEGSimpleConv, self).__init__()
        self.conv = torch.nn.Conv1d(n_chan, fm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)
        self.bn = torch.nn.BatchNorm1d(fm)
        self.num_channels = list(itertools.accumulate([
                fm if i==0 else( 1 if i==1 else (1.414)) for i in range(n_convs + 1)
            ],
            lambda x, y: int(x * y)
        ))

        self.blocks = nn.ModuleList([
            ConvLayer(
                kernel_size, old_fm, new_fm, 2
            )
            for i, (old_fm, new_fm) in enumerate(zip(self.num_channels[:-1], self.num_channels[1:]))
        ])

        if n_classes > 2:
            self.fc = torch.nn.Linear(self.num_channels[-1], n_classes)
        
        else:
            self.fc = nn.Sequential(
                torch.nn.Linear(self.num_channels[-1], 1),
                Rearrange('b 1 -> (b 1)'),    
            )

        self.fc2 = torch.nn.Linear(
            self.num_channels[-1], n_subjects
        ) if n_subjects else None #Subject regularization


    def forward(self, x: torch.Tensor):
        # x: (bsz, n_chan, time, freq)
        x = x.flatten(-2) # x: (bsz, n_chan, time * freq)
        y = torch.relu(self.bn(self.conv(x)))
        for seq in self.blocks:
            y = seq(y)
        y = y.mean(dim = 2)
        return (self.fc(y),self.fc2(y)) if self.fc2 else self.fc(y)

