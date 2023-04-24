import torch
from .position import PositionEmbeddingSine
import torch.nn.functional as F
from .geometry import generate_window_grid


def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [split_feature(x, num_splits=attn_splits) for x in features_list]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [merge_splits(x, num_splits=attn_splits) for x in features_splits]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def sample_features_by_grid(raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border',
                            local_radius=0, local_dilation=1):
    if local_radius <= 0:
        return F.grid_sample(raw_whole_feats, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)

    # --- sample on a local grid
    # unnomarlize original gird
    h, w = raw_whole_feats.shape[-2:]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(grid.device)  # inverse scale
    unnorm_grid = (grid * c + c).reshape(grid.shape[0], -1, 2)  # [B, n_rays*n_pts, 2]
    # build local grid
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=raw_whole_feats.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(1, -1, 2).repeat(grid.shape[0], 1, 1) * local_dilation  # [B, (2R+1)^2, 2]
    # merge grid and normalize
    sample_grid = unnorm_grid.unsqueeze(2) + window_grid.unsqueeze(1)  # [B, n_rays*n_pts, (2R+1)^2, 2]
    c = torch.Tensor([(w + local_w * local_dilation - 1) / 2.,
                    (h + local_h * local_dilation - 1) / 2.]).float().to(sample_grid.device)  # inverse scale
    norm_sample_grid = (sample_grid - c) / c  # range (-1, 1)
    # sample features
    sampled_feats = F.grid_sample(raw_whole_feats, norm_sample_grid,
                                align_corners=align_corners, mode=mode, padding_mode=padding_mode)  # [B, C, n_rays*n_pts, (2R+1)^2]
    # merge features of local grid
    b, c, n = sampled_feats.shape[:3]
    n_rays, n_pts = grid.shape[1:3]
    sampled_feats = sampled_feats.reshape(b, c*n, local_h, local_w)  # [B, C*n_rays*n_pts, 2R+1, 2R+1]
    avg_feats = F.adaptive_avg_pool2d(sampled_feats, (1, 1))  # [B, C*n_rays*n_pts, 1, 1]
    avg_feats = avg_feats.reshape(b, c, n_rays, n_pts)
    return avg_feats
