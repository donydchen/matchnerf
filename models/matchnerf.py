import torch
from easydict import EasyDict as edict
import torch.nn.functional as torch_F
from tqdm import tqdm

from .rfdecoder.cond_nerf import CondNeRF
from .gmflow.gmflow import GMFlow
from .gmflow.utils import sample_features_by_grid

from misc import camera


class MatchNeRF(torch.nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.nerf_setbg_opaque = False
        self.n_src_views = opts.n_src_views

        # init encoder
        feat_enc = GMFlow(feature_channels=128, num_scales=1, num_head=1,
                          attention_type='swin', ffn_dim_expansion=4,
                          feature_upsampler=opts.encoder.feature_upsampler,
                          upsample_factor=opts.encoder.upsample_factor,
                          num_transformer_layers=opts.encoder.num_transformer_layers,
                          device=opts.device).to(opts.device)
        self.feat_enc = feat_enc

        # define decoder
        self.nerf_dec = CondNeRF(opts).to(opts.device)

    def forward(self, batch, mode=None, render_video=False, render_path_mode='interpolate'):
        ref_images = batch.images[:, :self.n_src_views]
        # extract enhanced features, ref_images should be normalized by imagenet (mean, std)
        ref_feats_list = self.get_img_feat(ref_images, attn_splits_list=self.opts.encoder.attn_splits_list,
                                           cur_n_src_views=self.n_src_views)
        # reconstruct target and reference pose info from input batch
        tgt_pose, ref_poses = self.extract_poses(batch)

        batch_size, _, _, img_h, img_w = ref_images.shape

        if render_video:
            assert mode in ['test', 'val'], f"Do NOT render video in mode {mode}, change to either 'test' or 'val'."
            poses_paths = self.get_video_rendering_path(tgt_pose, ref_poses, render_path_mode, self.opts.nerf.video_n_frames, batch)
        else:
            poses_paths = [tgt_pose]

        # render images
        mode_rand_rays = getattr(self.opts.nerf, f'rand_rays_{mode}', 0)
        for frame_idx, cur_tgt_pose in enumerate(tqdm(poses_paths, desc="rendering video frame...", leave=False) if render_video else poses_paths):
            if mode_rand_rays and mode in ["train", "test-optim"]:
                # sample random rays for optimization
                batch.ray_idx = torch.randperm(img_h*img_w, device=self.opts.device)[:mode_rand_rays//batch_size]
                ret = self.render(self.opts, cur_tgt_pose, ray_idx=batch.ray_idx, mode=mode,
                                  ref_poses=ref_poses, ref_images=ref_images, ref_feats_list=ref_feats_list)  # [B,N,3],[B,N,1]
            else:
                # render full image (process in slices)
                if mode_rand_rays:
                    ret = self.render_by_slices(self.opts, cur_tgt_pose, mode=mode,
                                                ref_poses=ref_poses, ref_images=ref_images, ref_feats_list=ref_feats_list)
                else:
                    ret = self.render(self.opts, cur_tgt_pose, mode=mode,
                                      ref_poses=ref_poses, ref_images=ref_images, ref_feats_list=ref_feats_list)  # [B,HW,3],[B,HW,1]

            if frame_idx == 0:
                batch.update(edict({k: [] for k in ret.keys()}))
            for k, v in ret.items():
                batch[k].append(v.detach().cpu() if render_video else v)
            if frame_idx == len(poses_paths) - 1:
                for k in ret.keys():
                    batch[k] = torch.cat(batch[k], dim=0)

        return batch

    def extract_poses(self, batch):
        tgt_pose = {}
        tgt_pose['extrinsics'] = batch.extrinsics[:, -1, :3, :]  # B, 3, 4
        tgt_pose['intrinsics'] = batch.intrinsics[:, -1]  # B, 3, 3
        tgt_pose['near_fars'] = batch.near_fars[:, -1]  # B, 2

        ref_poses = {}
        ref_poses['extrinsics'] = batch.extrinsics[:, :-1, :3, :]  # B, N, 3, 4
        ref_poses['intrinsics'] = batch.intrinsics[:, :-1]  # B, N, 3, 3
        ref_poses['near_fars'] = batch.near_fars[:, :-1]  # B, N, 2

        return tgt_pose, ref_poses

    def render(self, opt, tgt_pose=None, ray_idx=None, mode=None,
               ref_poses=None, ref_images=None, ref_feats_list=None):
        """
            tgt_pose: dict, all camera information of the target viewpoint 
                        'extrinsics': B, 3, 4; 'intrinsics': B, 3, 3; 'near_fars': B, 2
            ref_poses: dict, all camera information of reference images 
                        'extrinsics': B, N, 3, 4; 'intrinsics': B, N, 3, 3; 'near_fars': B, N, 2
            ref_images: B, N, 3, H, W
            ref_feats_list: n_scales list,
        """
        # batch_size = len(tgt_pose['extrinsics'])
        batch_size, _, _, img_h, img_w = ref_images.shape
        if tgt_pose is None:
            raise Exception('Must provide tgt_pose.')

        # casting ray with the given camera parameters, in cannonical world sapce
        center, ray = camera.get_center_and_ray(img_h, img_w, tgt_pose['extrinsics'], intr=tgt_pose['intrinsics'],
                                                legacy=opt.nerf.legacy_coord, device=opt.device)  # [B,HW,3]
        while ray.isnan().any():  # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center, ray = camera.get_center_and_ray(img_h, img_w, tgt_pose['extrinsics'], intr=tgt_pose['intrinsics'],
                                                    legacy=opt.nerf.legacy_coord, device=opt.device)  # [B,HW,3]
        # consider only subset of rays
        if ray_idx is not None:
            center, ray = center[:, ray_idx], ray[:, ray_idx]
        depth_samples = self.sample_depth(opt, batch_size, num_rays=ray.shape[1], near_far=tgt_pose['near_fars'],
                                          legacy=opt.nerf.legacy_coord, mode=mode)  # [B,HW,N,1]
        # cast rays along depth range
        pts_3D = camera.get_3D_points_from_depth(opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,N,3]

        # query features from encoder
        cond_info = self.query_cond_info(pts_3D, ref_poses, ref_images, ref_feats_list)

        # Warp the input position to reference coordinate and convert to NDC, use the first image as reference
        coord_ref_idx = 0
        ref_extr = ref_poses['extrinsics'][:, coord_ref_idx]
        ref_intr = ref_poses['intrinsics'][:, coord_ref_idx]
        ref_nf = ref_poses['near_fars'][:, coord_ref_idx]
        inv_scale = torch.tensor([[img_w - 1, img_h - 1]]).repeat(batch_size, 1).to(opt.device)
        pts_3D_ref_ndc = camera.get_coord_ref_ndc(ref_extr, ref_intr, pts_3D, inv_scale, ref_nf)

        # query rgb and density from decoder
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1)
            ray_unit_ref = ray_unit @ ref_extr[..., :3, :3].transpose(-1, -2)
            ray_unit_ref = ray_unit_ref.unsqueeze(-2).repeat(1, 1, pts_3D_ref_ndc.shape[-2], 1)  # (B, n_rays, n_samples, 3)
        else:
            ray_unit_ref = None
        rgb_samples, density_samples = self.nerf_dec(self.opts, pts_3D_ref_ndc, ray_unit=ray_unit_ref, cond_info=cond_info)

        # volume rendering to get final 2D output
        comp_func = self.nerf_dec.module.composite if isinstance(self.nerf_dec, torch.nn.DataParallel) else self.nerf_dec.composite
        rgb, depth, opacity, prob = comp_func(self.opts, ray, rgb_samples, density_samples, depth_samples,
                                              setbg_opaque=self.nerf_setbg_opaque)
        ret = edict(rgb=rgb, depth=depth, opacity=opacity)  # [B,HW,K]

        return ret

    def render_by_slices(self, opt, tgt_pose, mode=None,
                         ref_poses=None, ref_images=None, ref_feats_list=None):
        assert ref_images is not None, "Must provide the reference images for MatchNeRF."
        img_h, img_w = ref_images.shape[-2:]
        ret_all = edict(rgb=[], depth=[], opacity=[])
        # render the image by slices for memory considerations
        mode_rand_rays = getattr(opt.nerf, f'rand_rays_{mode}', 0)
        for c in tqdm(range(0, img_h*img_w, mode_rand_rays), desc=f"slicing per [{mode_rand_rays}] rays...", leave=False):
            ray_idx = torch.arange(c, min(c+mode_rand_rays, img_h*img_w), device=opt.device)
            ret = self.render(opt, tgt_pose, ray_idx=ray_idx, mode=mode,
                              ref_poses=ref_poses, ref_images=ref_images, ref_feats_list=ref_feats_list)  # [B,R,3],[B,R,1]
            for k in ret:
                ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all:
            ret_all[k] = torch.cat(ret_all[k], dim=1)
        return ret_all

    def sample_depth(self, opt, batch_size, num_rays, near_far, legacy=False, mode='train'):
        depth_min, depth_max = torch.split(near_far, [1, 1], dim=-1)
        rand_shift = 0. if legacy else 0.5
        depth_denom = opt.nerf.sample_intvs - 1 if legacy else opt.nerf.sample_intvs

        if mode == 'train' and opt.nerf.sample_stratified:
            rand_samples = torch.rand(batch_size, num_rays, opt.nerf.sample_intvs, 1, device=opt.device)
        else:
            rand_samples = rand_shift * torch.ones(batch_size, num_rays, opt.nerf.sample_intvs, 1, device=opt.device)

        rand_samples = rand_samples + torch.arange(opt.nerf.sample_intvs, device=opt.device)[None, None, :, None].float()  # [B,HW,N,1]
        depth_max = depth_max.reshape(batch_size, *[1]*(rand_samples.dim() - 1))
        depth_min = depth_min.reshape(batch_size, *[1]*(rand_samples.dim() - 1))
        depth_samples = rand_samples / depth_denom * (depth_max - depth_min) + depth_min  # [B,HW,N,1]  # for +0.5
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def get_img_feat(self, imgs, attn_splits_list=None, cur_n_src_views=3):
        if attn_splits_list is None:
            attn_splits_list = self.opts.encoder.attn_splits_list
        img1s = imgs[:, :cur_n_src_views]

        # run gmflow backbone to extract features
        out_dict = self.feat_enc(imgs=img1s, attn_splits_list=attn_splits_list, keep_raw_feats=True,
                                 wo_self_attn=self.opts.encoder.wo_self_attn)
        # split the output
        img_feat_list = []
        index_lists = [(a, b) for a in range(cur_n_src_views - 1) for b in range(a + 1, cur_n_src_views)]
        for scale_idx in range(len(out_dict['aug_feat0s'])):
            img_feat = [[] for _ in range(cur_n_src_views)]
            img1s_feats = out_dict['aug_feat0s'][scale_idx]
            img2s_feats = out_dict['aug_feat1s'][scale_idx]
            for feat_i, (i_idx, j_idx) in enumerate(index_lists):
                img_feat[i_idx].append(img1s_feats[:, feat_i])
                img_feat[j_idx].append(img2s_feats[:, feat_i])
            # post-process the output
            for k, v in enumerate(img_feat):
                img_feat[k] = torch.cat(v, dim=1)
            img_feat = torch.stack(img_feat, dim=1)  # BxVxCxHxW
            img_feat_list.append(img_feat)

        return img_feat_list

    def query_cond_info(self, point_samples, ref_poses, ref_images, ref_feats_list):
        '''
            query conditional information from reference images, using the target position.
            point_samples: B, n_rays, n_samples, 3
            ref_poses: dict, all camera information of reference images 
                        'extrinsics': B, N, 3, 4; 'intrinsics': B, N, 3, 3; 'near_fars': B, N, 2
            ref_images: B, n_views, 3, H, W. range: [0, 1] !!!
        '''
        batch_size, n_views, _, img_h, img_w = ref_images.shape
        assert ref_feats_list is not None, "Must provide the image feature for info query."

        device = self.opts.device
        cos_n_group = self.opts.encoder.cos_n_group
        cos_n_group = [cos_n_group] if isinstance(cos_n_group, int) else cos_n_group
        feat_data_list = [[] for _ in range(len(ref_feats_list))]
        color_data = []
        mask_data = []

        # query information from each source view
        inv_scale = torch.tensor([[img_w - 1, img_h - 1]]).repeat(batch_size, 1).to(device)
        for view_idx in range(n_views):
            near_far_ref = ref_poses['near_fars'][:, view_idx]
            extr_ref, intr_ref = ref_poses['extrinsics'][:, view_idx].clone(), ref_poses['intrinsics'][:, view_idx].clone()
            point_samples_pixel = camera.get_coord_ref_ndc(extr_ref, intr_ref, point_samples,
                                                           inv_scale, near_far=near_far_ref)
            grid = point_samples_pixel[..., :2] * 2.0 - 1.0

            # query enhanced features infomation from each view
            for scale_idx, img_feat_cur_scale in enumerate(ref_feats_list):
                raw_whole_feats = img_feat_cur_scale[:, view_idx]
                sampled_feats = sample_features_by_grid(raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border',
                                                        local_radius=self.opts.encoder.feature_sample_local_radius,
                                                        local_dilation=self.opts.encoder.feature_sample_local_dilation)
                feat_data_list[scale_idx].append(sampled_feats)

            # query color
            color_data.append(torch_F.grid_sample(ref_images[:, view_idx], grid, align_corners=True, mode='bilinear', padding_mode='border'))

            # record visibility mask for further usage
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1]).float()
            mask_data.append(in_mask.unsqueeze(1))

        # merge queried information from all views
        all_data = {}
        # merge extracted enhanced features
        merged_feat_data = []
        for feat_data_idx, raw_feat_data in enumerate(feat_data_list):  # loop over scale
            cur_updated_feat_data = []
            # split back to original
            split_feat_data = [torch.split(x, int(x.shape[1] / (n_views - 1)), dim=1) for x in raw_feat_data]
            # calculate simliarity for feature from the same transformer
            index_lists = [(a, b) for a in range(n_views - 1) for b in range(a, n_views - 1)]
            for i_idx, j_idx in index_lists:
                input_a = split_feat_data[i_idx][j_idx]  # B x C x N_rays x N_pts
                input_b = split_feat_data[j_idx + 1][i_idx]
                iB, iC, iR, iP = input_a.shape
                group_a = input_a.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
                group_b = input_b.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
                cur_updated_feat_data.append(torch.nn.CosineSimilarity(dim=2)(group_a, group_b))
            cur_updated_feat_data = torch.stack(cur_updated_feat_data, dim=1)  # [B, n_pairs, n_groups, n_rays, n_pts]

            cur_updated_feat_data = torch.mean(cur_updated_feat_data, dim=1, keepdim=True)
            cur_updated_feat_data = cur_updated_feat_data.reshape(cur_updated_feat_data.shape[0], -1, *cur_updated_feat_data.shape[-2:])
            merged_feat_data.append(cur_updated_feat_data)

        merged_feat_data = torch.cat(merged_feat_data, dim=1)
        # all_data.append(merged_feat_data)
        all_data['feat_info'] = merged_feat_data

        # merge extracted color data
        merged_color_data = torch.cat(color_data, dim=1)
        # all_data.append(merged_color_data)
        all_data['color_info'] = merged_color_data

        # merge visibility masks
        merged_mask_data = torch.cat(mask_data, dim=1)
        # all_data.append(merged_mask_data)
        all_data['mask_info'] = merged_mask_data

        # all_data = torch.cat(all_data, dim=1)[0].permute(1, 2, 0)
        for k, v in all_data.items():
            all_data[k] = v.permute(0, 2, 3, 1)  # (b, n_rays, n_samples, n_dim)

        return all_data

    def get_video_rendering_path(self, tgt_pose, ref_poses, mode, n_frames=30, batch=None):
        # loop over batch
        poses_paths = []
        for batch_idx, cur_src_poses in enumerate(ref_poses['extrinsics']):
            if mode == 'interpolate':
                # convert to c2ws
                pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1).to(self.opts.device)
                pose_square[:, :3, :] = cur_src_poses
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
                cur_path = camera.get_interpolate_render_path(cur_c2ws, n_frames)
            elif mode == 'spiral':
                assert batch is not None, "Must provide all c2ws and near_far for getting spiral rendering path."
                cur_c2ws_all = batch['c2ws_all'][batch_idx].detach().cpu().numpy()
                cur_near_far = tgt_pose['near_fars'][batch_idx].detach().cpu().numpy().tolist()
                rads_scale = getattr(self.opts.nerf, "video_rads_scale", 0.1)
                cur_path = camera.get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32).to(self.opts.device)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        updated_tgt_poses = []
        for frame_idx in range(n_frames):
            updated_tgt_poses.append(dict(extrinsics=poses_paths[:, frame_idx],
                                          intrinsics=tgt_pose['intrinsics'].clone().detach(),
                                          near_fars=tgt_pose['near_fars'].clone().detach()))

        return updated_tgt_poses
