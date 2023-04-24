from .llff import MVSDatasetRealFF
import torch
import os
from PIL import Image
import numpy as np
from glob import glob
from pathlib import Path
from misc.utils import list_all_images
from .llff import center_poses


def gen_pairs(root_dir, n_select=20, n_interval=6):
    ''' TODO: mainly for data from LLFF project. '''
    subdirs = glob(os.path.join(root_dir, '*/'))
    n_select = int(n_select)
    n_interval = int(n_interval)
    pairs = {}
    for subdir in subdirs:
        scene = os.path.basename(subdir.strip('/'))
        pose_meta_file = os.path.join(subdir, 'poses_bounds.npy')
        assert os.path.isfile(pose_meta_file), f"Please run COLMAP for {subdir} first, using imgs2pose from LLFF project."

        poses_bounds = np.load(pose_meta_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        N_images = poses.shape[0]
        if N_images <= 3:
            pairs[f'{scene}_test'] = np.array([0])
            pairs[f'{scene}_val'] = np.array([0])
            pairs[f'{scene}_train'] = np.array([2, 1, 0])  # np.array(list(range(N_images)))
            continue

        # print(subdir, 'N_images', N_images)
        n_select_safe = min(N_images, n_select)
        n_interval_safe = min(N_images, n_interval)
        # correct pose, from [down right back] to [left up back]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

        ref_position = np.mean(poses[..., 3], axis=0, keepdims=True)  # use the center camera as reference
        dist = np.sum(np.abs(poses[..., 3] - ref_position), axis=-1)
        pair_idx = np.argsort(dist)[:n_select_safe]

        pairs[f'{scene}_test'] = pair_idx[::n_interval_safe]
        pairs[f'{scene}_val'] = pair_idx[::n_interval_safe]
        pairs[f'{scene}_train'] = np.delete(pair_idx, range(0, n_select_safe, n_interval_safe))
    return pairs


class MVSDatasetCOLMAP(MVSDatasetRealFF):
    """docstring for MVSDatasetColMap."""

    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 scene_list=None, test_views_method='nearest', nf_mode='avg', **kwargs):
        ''' TODO: Notice that this is currently dedicated to forward facing data. '''
        assert split in ['test'], 'Only support "test" split for blender dataset!'

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.max_len = max_len
        self.nf_mode = nf_mode
        self.eval_mode = 'mvsnerf'

        self.img_wh = img_wh
        # if img_wh is not None:
        # assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'
        self.transform = self.define_transforms()

        if scene_list is None:
            scene_list = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        pairs_dict = gen_pairs(root_dir, 20, 6)
        # print(pairs_dict)
        # assert False
        if test_views_method == 'fixed':  # currently only consider video rendering
            for k, v in pairs_dict.items():
                if k.split('_')[-1] == 'val':
                    pairs_dict[k] = v[:1]
        self.metas, self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, \
            self.near_fars_dict, self.imgs_paths_dict = self.build_test_metas(scene_list, pairs_dict, method=test_views_method)

    def get_name(self):
        dataname = 'colmap'
        return dataname

    def build_camera_info_per_scene(self, id_list, meta_filepath, scene_name):
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        poses_bounds = np.load(meta_filepath)  # (N_images, 17)

        images_dir = os.path.join(Path(meta_filepath).parent.absolute(), 'images')
        images_list = list_all_images(images_dir)

        poses = poses_bounds[:, :15].copy().reshape(-1, 3, 5)  # (N_images, 3, 5)
        # correct pose, from [down right back] to [left up back]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # poses = center_poses(poses, blender2opencv)
        poses = poses @ blender2opencv  # no need to center poses, matchnerf use relative coordinate system.

        # raw near far bounds
        bounds = poses_bounds[:, -2:].copy()  # (N_images, 2)

        # correct scale so that the nearest depth is at a little more than 1.0
        near_original = bounds.min()
        scale_factor = near_original * 0.47058824  # 0.75  # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        intrinsics, world2cams, cam2worlds, near_fars, imgs_paths = {}, {}, {}, {}, {}
        w, h = self.img_wh
        for view_idx in id_list:
            # intrinsic
            raw_h, raw_w, focal = poses_bounds[:, :15].copy().reshape(-1, 3, 5)[view_idx, :, -1]  # original intrinsics
            intr = np.array([[focal * w / raw_w, 0, w / 2],
                            [0, focal * h / raw_h, h / 2],
                            [0, 0, 1]])
            intrinsics[f'{scene_name}_{view_idx}'] = intr

            c2w = np.eye(4)
            c2w[:3] = poses[view_idx]
            cam2worlds[f'{scene_name}_{view_idx}'] = c2w  # 4x4

            # original codebase use torch to get inverse matrix, here match the dtype as float32
            w2c = np.linalg.inv(c2w.astype(np.float32))
            world2cams[f'{scene_name}_{view_idx}'] = w2c

            near_fars[f'{scene_name}_{view_idx}'] = bounds[view_idx]

            imgs_paths[f'{scene_name}_{view_idx}'] = images_list[view_idx]

        return intrinsics, world2cams, cam2worlds, near_fars, imgs_paths

    def __getitem__(self, idx):
        sample = {}

        scene, target_view, src_views, ori_train_views = self.metas[idx]
        view_ids = [src_views[i] for i in range(self.n_views)] + [target_view]

        imgs, intrinsics, w2cs, near_fars = [], [], [], []
        img_wh = np.array(self.img_wh).astype('int')
        for vid in view_ids:
            img_filename = os.path.join(self.root_dir, scene, 'images', self.imgs_paths_dict[f'{scene}_{vid}'])
            img = Image.open(img_filename)
            img = img.resize(img_wh, Image.LANCZOS)
            img = self.transform(img)
            imgs.append(img)

            intrinsics.append(self.intrinsics_dict[f'{scene}_{vid}'])
            w2cs.append(self.world2cams_dict[f'{scene}_{vid}'])
            near_fars.append(self.near_fars_dict[f'{scene}_{vid}'])

        sample['images'] = torch.stack(imgs).float()  # (V, H, W, 3)
        sample['extrinsics'] = np.stack(w2cs).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack(intrinsics).astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['scene'] = scene
        sample['img_wh'] = img_wh

        # COLMAP data has different near_far for different views, reset them to be the same to better fit the pretrained model
        # print(np.stack(near_fars))
        if self.nf_mode == 'minmax':
            # print('use_minmax')
            near_fars = np.stack(near_fars)
            sample['near_fars'] = np.expand_dims(np.array([near_fars.min() * 0.8, near_fars.max() * 1.2]), axis=0).repeat(len(view_ids), axis=0).astype(np.float32)
        elif self.nf_mode == 'avg':
            # print('use_avg')
            sample['near_fars'] = np.expand_dims(np.average(np.stack(near_fars), axis=0), axis=0).repeat(len(view_ids), axis=0).astype(np.float32)
        else:
            raise Exception(f'Unknown near far mode {self.nf_mode}')

        # c2ws for all train views, required for rendering videos
        c2ws_all = [self.cam2worlds_dict[f'{scene}_{x}'] for x in ori_train_views]
        sample['c2ws_all'] = np.stack(c2ws_all).astype(np.float32)
        # print(sample['near_fars'])

        return sample
