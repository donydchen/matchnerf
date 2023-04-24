import torch.nn as nn
from math import log2


class UpSampler(nn.Module):
    ''' UpSampler class
    '''

    def __init__(self, n_feat=128, upsample_factor=8, **kwargs):
        super().__init__()
        n_blocks = int(log2(upsample_factor))
        self.n_blocks = n_blocks

        self.upsample_l = nn.Upsample(scale_factor=2.)
        self.upsample_r = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.conv_ls = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat, 3, 1, 1) for _ in range(n_blocks)]
        )
        self.conv_l2rs = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat, 3, 1, 1) for _ in range(n_blocks + 1)]
        )

        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        right_x = self.conv_l2rs[0](x)
        left_x = x

        for idx in range(self.n_blocks):
            # left branch
            left_x = self.actvn(self.conv_ls[idx](self.upsample_l(left_x)))
            # left to right
            mid_x = self.conv_l2rs[idx + 1](left_x)
            # right branch
            right_x = self.upsample_r(right_x) + mid_x

        return right_x
