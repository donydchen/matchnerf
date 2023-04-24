from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from pathlib import Path
from misc.utils import list_all_images


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered


class MVSDatasetRealFF(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 scene_list=None, test_views_method='nearest', eval_mode='mvsnerf', **kwargs):
        assert split in ['test'], 'Only support "test" split for blender dataset!'

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.max_len = max_len
        self.eval_mode = eval_mode
        if eval_mode == 'gpnr':
            self.test_hold_out = 8  # follow GPNR settings

        self.img_wh = img_wh
        # if img_wh is not None:
        # assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'
        self.transform = self.define_transforms()

        if scene_list is None:
            scene_list = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        pairs_dict = torch.load(os.path.join('configs', 'pairs.th'))
        self.metas, self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, \
            self.near_fars_dict, self.imgs_paths_dict = self.build_test_metas(scene_list, pairs_dict, method=test_views_method)

    def get_name(self):
        dataname = 'llff'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor(),])  # (3, h, w)
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    def build_test_metas(self, scene_list, pairs_dict, method='nearest'):
        metas = []
        intrinsics_dict, world2cams_dict, cam2worlds_dict, near_fars_dict = {}, {}, {}, {}
        imgs_paths_dict = {}

        # loop over all scene
        for cur_scene in scene_list:
            if self.eval_mode == 'mvsnerf':
                # follow MVSNeRF, test use 'val' split, train use 'train' split
                train_views = pairs_dict[f'{cur_scene}_train']
                test_views = pairs_dict[f'{cur_scene}_val']
            elif self.eval_mode == 'gpnr':
                # follow GPNR, use hold out mode to eval llff
                images_dir = os.path.join(self.root_dir, cur_scene, 'images')
                images_num = len(list_all_images(images_dir))
                test_views = np.arange(0, images_num, self.test_hold_out)
                train_views = np.array([x for x in range(images_num) if x not in test_views])
            else:
                Exception(f'Unknown eval_mode {self.eval_mode}.')

            cur_scene_info = self.build_test_metas_per_scene(cur_scene, train_views, test_views, method)
            metas.extend(cur_scene_info[0])
            intrinsics_dict.update(cur_scene_info[1])
            world2cams_dict.update(cur_scene_info[2])
            cam2worlds_dict.update(cur_scene_info[3])
            near_fars_dict.update(cur_scene_info[4])
            imgs_paths_dict.update(cur_scene_info[5])

        return metas, intrinsics_dict, world2cams_dict, cam2worlds_dict, near_fars_dict, imgs_paths_dict

    def build_test_metas_per_scene(self, scene_name, train_views, test_views, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []

        id_list = [*train_views, *test_views]
        meta_filepath = os.path.join(self.root_dir, scene_name, "poses_bounds.npy")

        intrinsics, world2cams, cam2worlds, \
            near_fars, imgs_paths = self.build_camera_info_per_scene(id_list, meta_filepath, scene_name)

        for target_view in test_views:
            # sort the reference source view accordingly
            if method == "nearest":
                cam_pos_trains = np.stack([cam2worlds[f'{scene_name}_{x}'] for x in train_views])[:, :3, 3]
                cam_pos_target = cam2worlds[f'{scene_name}_{target_view}'][:3, 3]
                dis = np.sum(np.abs(cam_pos_trains - cam_pos_target), axis=-1)
                src_idx = np.argsort(dis)
                src_idx = [train_views[x] for x in src_idx]
            elif method == "fixed":
                src_idx = train_views
            else:
                raise Exception('Unknown evaluate method [%s]' % method)

            metas.append((scene_name, target_view, src_idx, train_views))

        return metas, intrinsics, world2cams, cam2worlds, near_fars, imgs_paths

    def build_camera_info_per_scene(self, id_list, meta_filepath, scene_name):
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        poses_bounds = np.load(meta_filepath)  # (N_images, 17)

        images_dir = os.path.join(Path(meta_filepath).parent.absolute(), 'images')
        images_list = list_all_images(images_dir)

        poses = poses_bounds[:, :15].copy().reshape(-1, 3, 5)  # (N_images, 3, 5)
        # correct pose, from [down right back] to [left up back]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses = center_poses(poses, blender2opencv)

        # raw near far bounds
        bounds = poses_bounds[:, -2:].copy()  # (N_images, 2)

        # correct scale so that the nearest depth is at a little more than 1.0
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
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

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

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

        # LLFF has different near_far for different views, reset them to be the same to better fit the pretrained model
        sample['near_fars'] = np.expand_dims(np.average(np.stack(near_fars), axis=0), axis=0).repeat(len(view_ids), axis=0).astype(np.float32)

        # c2ws for all train views, required for rendering videos
        c2ws_all = [self.cam2worlds_dict[f'{scene}_{x}'] for x in ori_train_views]
        sample['c2ws_all'] = np.stack(c2ws_all).astype(np.float32)

        return sample
