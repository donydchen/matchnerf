import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .utils import feature_add_position
from .superres import UpSampler


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 feature_upsampler='none',
                 device='cuda',
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()
        self.device = device
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.feature_upsampler = feature_upsampler

        # CNN backbone, Hack the num_output_scales to support 1/4
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                            d_model=feature_channels,
                                            nhead=num_head,
                                            attention_type=attention_type,
                                            ffn_dim_expansion=ffn_dim_expansion,
                                            )

        if self.feature_upsampler == 'network':
            self.featup_net = UpSampler(n_feat=feature_channels, upsample_factor=upsample_factor)

    def extract_feature(self, imgs):
        b, n_views, c, h, w = imgs.shape
        index_lists = [(a, b) for a in range(n_views - 1) for b in range(a + 1, n_views)]

        features = self.backbone(imgs.reshape(b * n_views, c, h, w).to(imgs.device))  # list of [nB, C, H, W], resolution from high to low
        # reverse: resolution from low to high
        features = features[::-1]
        feature0_list, feature1_list = [], []
        for i in range(len(features)):
            feature = features[i]  # (b * n_views, n_feat_dim, h_i, w_i)
            feature = feature.reshape(b, int(feature.shape[0] / b), *feature.shape[-3:])
            cur_feat0, cur_feat1 = [], []
            for i_idx, j_idx in index_lists:
                cur_feat0.append(feature[:, i_idx, ...])
                cur_feat1.append(feature[:, j_idx, ...])
            cur_feat0 = torch.stack(cur_feat0, dim=1)
            cur_feat1 = torch.stack(cur_feat1, dim=1)
            feature0_list.append(cur_feat0.reshape(-1, *cur_feat0.shape[-3:]))
            feature1_list.append(cur_feat1.reshape(-1, *cur_feat1.shape[-3:]))

        return feature0_list, feature1_list

    def upsample_features(self, feature0, feature1):
        if self.feature_upsampler == 'none':
            return feature0, feature1

        merge_feats = torch.cat((feature0, feature1), dim=0)
        if self.feature_upsampler == 'network':
            up_merged_feats = self.featup_net(merge_feats)
        else:
            raise Exception('Unknown feature upsampler %s' % self.feature_upsampler)

        up_feature0, up_feature1 = torch.chunk(up_merged_feats, 2, 0)
        return up_feature0, up_feature1

    def normalize_images(self, images):
        '''to normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)'''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def forward(self, imgs, attn_splits_list=None, layers_caps=None, keep_raw_feats=False, wo_cross_attn=False,
                    wo_self_attn=False, **kwargs):
            ''' imgs: range [0, 1] '''
            results_dict = {}
            aug_feat0_list = []
            aug_feat1_list = []

            # extract backbone features
            batch_size, n_views, img_c, img_h, img_w = imgs.shape
            if img_h == 756 and img_w == 1008:  # ibrnet settings
                infer_size = (768, 1024)   # must be devisable by 16
                imgs = F.interpolate(imgs.reshape(batch_size * n_views, img_c, img_h, img_w),
                                    size=infer_size, mode='bilinear', align_corners=True).reshape(batch_size, n_views, img_c, *infer_size)

            feature0_list, feature1_list = self.extract_feature(self.normalize_images(imgs))
            
            # forward through the transformer structure
            all_scales = list(range(self.num_scales))
            if len(all_scales) != len(attn_splits_list):
                last_value = all_scales[-1]
                for _ in range(len(attn_splits_list) - len(all_scales)):
                    all_scales.append(last_value)
                
            if keep_raw_feats:
                assert len(all_scales) <= 1, "Only supports one scale for features up-sampler."
                assert self.feature_upsampler != 'none', "Must upsample the feature if keep_raw_feats."

            if layers_caps is not None:
                assert isinstance(layers_caps, list) or isinstance(layers_caps, tuple), '`layers_caps` must be list / tuple'
                assert len(layers_caps) == len(attn_splits_list), '`layers_cap` must have the same length as `attn_splits_list`.'
            else:
                layers_caps = [None] * len(attn_splits_list)

            for att_idx, scale_idx in enumerate(all_scales):
                # load backbone features
                feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

                attn_splits = attn_splits_list[att_idx]
                layers_cap = layers_caps[att_idx]

                # add position to features
                feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

                # Transformer
                feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits, layers_cap=layers_cap, wo_cross_attn=wo_cross_attn, wo_self_attn=wo_self_attn)

                # upsample sample feature map if specified
                if keep_raw_feats:
                    aug_feat0_list.append(feature0.reshape(batch_size, int(feature0.shape[0] / batch_size), *feature0.shape[-3:]))
                    aug_feat1_list.append(feature1.reshape(batch_size, int(feature1.shape[0] / batch_size), *feature1.shape[-3:]))

                feature0, feature1 = self.upsample_features(feature0, feature1)
                aug_feat0_list.append(feature0.reshape(batch_size, int(feature0.shape[0] / batch_size), *feature0.shape[-3:]))
                aug_feat1_list.append(feature1.reshape(batch_size, int(feature1.shape[0] / batch_size), *feature1.shape[-3:]))

            results_dict.update({
                'aug_feat0s': aug_feat0_list,
                'aug_feat1s': aug_feat1_list,})

            return results_dict
