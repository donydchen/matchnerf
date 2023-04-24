from .nerf import NeRF
import torch
from .ray_transformer import MultiHeadAttention
import torch.nn.functional as F
import numpy as np


class CondNeRF(NeRF):
    """Conditional NeRF; take (position, image_features) as input, output rgb images."""

    def __init__(self, opt):
        self.device = opt.device
        super(CondNeRF, self).__init__(opt)

    def define_network(self, opt):
        W = opt.decoder.net_width
        D = opt.decoder.net_depth
        input_ch_feat = sum(opt.encoder.cos_n_group) + opt.n_src_views * (3 + 1)
        input_3D_dim = 3 + 6 * opt.decoder.posenc.L_3D if opt.decoder.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3 + 6 * opt.decoder.posenc.L_view if opt.decoder.posenc else 3

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_3D_dim, W, bias=True)] +
            [torch.nn.Linear(W, W, bias=True) if i not in opt.decoder.skip else
             torch.nn.Linear(W + input_3D_dim, W) for i in range(D - 1)])
        self.pts_bias = torch.nn.Linear(input_ch_feat, W)

        if opt.nerf.view_dep:
            raytrans_act = getattr(torch.nn, opt.decoder.raytrans_act)(inplace=True)
            self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_view_dim + W, W // 2)])
            self.alpha_linear = torch.nn.Sequential(torch.nn.Linear(W, 16), raytrans_act)
            self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
            self.out_alpha_linear = torch.nn.Sequential(torch.nn.Linear(16, 16),
                                                        raytrans_act,
                                                        torch.nn.Linear(16, 1),
                                                        torch.nn.ReLU(inplace=True))
            if opt.decoder.raytrans_posenc:
                self.pos_encoding = self.posenc(d_hid=16, n_samples=opt.nerf.sample_intvs)

            self.feature_linear = torch.nn.Linear(W, W)
            self.rgb_linear = torch.nn.Linear(W // 2, 3)
        else:
            self.output_linear = torch.nn.Linear(W, 3 + 1)

        self.pts_linears.apply(self.weights_init)
        self.views_linears.apply(self.weights_init)
        self.feature_linear.apply(self.weights_init)
        self.alpha_linear.apply(self.weights_init)
        self.rgb_linear.apply(self.weights_init)

    def forward(self, opt, points_3D, ray_unit=None, cond_info=None, mode=None):
        pos_enc_func = self.positional_encoding_legacy if opt.nerf.legacy_coord else self.positional_encoding
        if opt.decoder.posenc:
            points_enc = pos_enc_func(opt, points_3D, L=opt.decoder.posenc.L_3D)
            points_enc = torch.cat((points_3D, points_enc), dim=-1)  # [B, n_rays, n_samples, 6L+3]
        else:
            points_enc = points_3D  # [B, n_rays, n_samples, 3]
        input_feats = torch.cat([cond_info['feat_info'], cond_info['color_info'], cond_info['mask_info']], dim=-1)

        h = points_enc
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = l(h) * bias
            h = F.relu(h)
            if i in opt.decoder.skip:
                h = torch.cat([points_enc, h], -1)

        if opt.nerf.view_dep:
            if opt.decoder.posenc and opt.decoder.posenc.L_view > 0:
                ray_enc = pos_enc_func(opt, ray_unit, L=opt.decoder.posenc.L_view)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            else:
                ray_enc = ray_unit

            raw_alpha = self.alpha_linear(h)  # [B, n_rays, n_samples, 16]
            if opt.decoder.raytrans_posenc and hasattr(self, 'pos_encoding'):
                raw_alpha = raw_alpha + self.pos_encoding.to(raw_alpha.device)
            mask = cond_info['mask_info']  # [B, n_rays, n_samples, n_views]
            num_valid_obs = torch.sum(mask, dim=-1, keepdim=True)  # [B, n_rays, n_samples, 1]
            raw_alpha = raw_alpha.reshape(-1, *raw_alpha.shape[-2:])  # [Bxn_rays, n_samples, 16]
            num_valid_obs = num_valid_obs.reshape(-1, *num_valid_obs.shape[-2:])  # [Bxn_rays, n_samples, 1]
            alpha, _ = self.ray_attention(raw_alpha, raw_alpha, raw_alpha,
                                          mask=(num_valid_obs > 1).float())  # [Bxn_rays, n_samples, 16]
            alpha = self.out_alpha_linear(alpha)
            if opt.decoder.density_maskfill:
                alpha = alpha.masked_fill(num_valid_obs < 1, 0.)
            alpha = alpha.reshape(*points_enc.shape[:-1])  # [B, n_rays, n_samples]

            feature = self.feature_linear(h)
            h = torch.cat([feature, ray_enc], -1)
            for i, l in enumerate(self.views_linears):
                h = l(h)
                h = F.relu(h)
            rgb = torch.sigmoid(self.rgb_linear(h))  # [B, n_rays, n_samples, 3]
        else:
            rgb, alpha = torch.split(self.output_linear(h), [3, 1], dim=-1)
            alpha = alpha.unsqueeze(-1)

        return rgb, alpha

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

    def positional_encoding_legacy(self, opt, input, L):  # [B,...,N]
        '''frequency without `pi`, and to match the dimension order with the origianl codebase'''
        shape = input.shape  # (B, n_rays, n_samples, 3)
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device)  # [L]
        spectrum = input.unsqueeze(-2) * freq.reshape(*[1]*(len(shape) - 1), -1, 1)
        spectrum = spectrum.reshape(*shape[:-1], -1)
        input_enc = torch.cat((spectrum.sin(), spectrum.cos()), dim=-1)

        return input_enc

    def posenc(self, d_hid, n_samples):
        '''ray transformer position encoding'''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to(self.device).float().unsqueeze(0)
        return sinusoid_table
