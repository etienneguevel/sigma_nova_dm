import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from sigma_nova_dm.models.cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param, device):
        super(Model, self).__init__()

        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            print(f"loading state dict from {param.foundation_dir}")
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=device))
        
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out
