import torch
import torch.nn.functional as torch_F
from misc import camera, utils
import numpy as np


class NeRF(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self, opt):
        input_3D_dim = 3+6*opt.decoder.posenc.L_3D if opt.decoder.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.decoder.posenc.L_view if opt.decoder.posenc else 3
        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = utils.get_layer_dims(opt.decoder.layers_feat)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = input_3D_dim
            if li in opt.decoder.skip:
                k_in += input_3D_dim
            if li == len(L)-1:
                k_out += 1
            linear = torch.nn.Linear(k_in, k_out)
            if opt.decoder.tf_init:
                self.tensorflow_init_weights(opt, linear, out="first" if li == len(L)-1 else None)
            self.mlp_feat.append(linear)
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = utils.get_layer_dims(opt.decoder.layers_rgb)
        feat_dim = opt.decoder.layers_feat[-1]
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in, k_out)
            if opt.decoder.tf_init:
                self.tensorflow_init_weights(opt, linear, out="all" if li == len(L)-1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self, opt, linear, out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu")  # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out == "first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:], gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, opt, points_3D, ray_unit=None, mode=None, **kwargs):  # [B,...,3]
        if opt.decoder.posenc:
            points_enc = self.positional_encoding(opt, points_3D, L=opt.decoder.posenc.L_3D)
            points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D
        feat = points_enc
        # extract coordinate-based features
        for li, layer in enumerate(self.mlp_feat):
            if li in opt.decoder.skip:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li == len(self.mlp_feat)-1:
                density = feat[..., 0]
                if opt.nerf.density_noise_reg and mode == "train":
                    density += torch.randn_like(density)*opt.nerf.density_noise_reg
                density_activ = getattr(torch_F, opt.decoder.density_activ)  # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[..., 1:]
            feat = torch_F.relu(feat)
        # predict RGB values
        if opt.nerf.view_dep:
            assert(ray_unit is not None)
            if opt.decoder.posenc:
                ray_enc = self.positional_encoding(opt, ray_unit, L=opt.decoder.posenc.L_view)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            else:
                ray_enc = ray_unit
            feat = torch.cat([feat, ray_enc], dim=-1)
        for li, layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li != len(self.mlp_rgb)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_()  # [B,...,3]
        return rgb, density

    def forward_samples(self, opt, center, ray, depth_samples, mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,HW,3]
            ray_unit_samples = ray_unit[..., None, :].expand_as(points_3D_samples)  # [B,HW,N,3]
        else:
            ray_unit_samples = None
        rgb_samples, density_samples = self.forward(opt, points_3D_samples, ray_unit=ray_unit_samples, mode=mode)  # [B,HW,N],[B,HW,N,3]
        return rgb_samples, density_samples

    def composite(self, opt, ray, rgb_samples, density_samples, depth_samples, setbg_opaque):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0]-depth_samples[..., :-1, 0]  # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)], dim=2)  # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length  # [B,HW,N]
        if opt.nerf.wo_render_interval:
            # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
            # very different scales, and using interval can affect the model's generalization ability.
            # Therefore we don't use the intervals for both training and evaluation. [IBRNet]
            sigma_delta = density_samples
        else:
            sigma_delta = density_samples*dist_samples  # [B,HW,N]

        alpha = 1-(-sigma_delta).exp_()  # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2).cumsum(dim=2)).exp_()  # [B,HW,N]
        prob = (T*alpha)[..., None]  # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2)  # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2)  # [B,HW,3]
        opacity = prob.sum(dim=2)  # [B,HW,1]
        if setbg_opaque:
            rgb = rgb + 1 * (1 - opacity)
        return rgb, depth, opacity, prob  # [B,HW,K]

    def positional_encoding(self, opt, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi  # [L]
        spectrum = input[..., None]*freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc
