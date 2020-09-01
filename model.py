'''
    Pretrained Model Loader
    Source: https://github.com/lukemelas/EfficientNet-PyTorch.git
'''

import torch
import torch.nn as nn
from efficientnet_pytorch import model as enet


class enetv2(nn.Module):
    def __init__(self, backbone, backbone_pretrain, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.enet.load_state_dict(torch.load(backbone_pretrain))

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x