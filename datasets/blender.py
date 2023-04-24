from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import json


class MVSDatasetBlender(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 scene_list=None, test_views_method='nearest', eval_mode='mvsnerf', **kwargs):
        assert split in ['test'], 'Only support "test" split for blender dataset!'
        assert eval_mode in ['mvsnerf', 'gpnr'], "Only support mvsnerf and gpnr test mode."

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.max_len = max_len
        self.eval_mode = eval_mode

        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'
        self.transform = self.define_transforms()

        if scene_list is None:
            scene_list = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        pairs_dict = torch.load(os.path.join('configs', 'pairs.th'))
        self.metas, self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, \
            self.near_fars_dict, self.imgs_paths_dict = self.build_test_metas(scene_list, pairs_dict, method=test_views_method)

    def get_name(self):
        dataname = 'blender'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor(),  # (4, h, w), RGBA
                               T.Lambda(lambda x: x[:3] * x[-1:] + (1 - x[-1:])),])  # blend A to RGB
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    def build_test_metas(self, scene_list, pairs_dict, method='nearest'):
        metas = []
        intrinsics_dict, world2cams_dict, cam2worlds_dict, near_fars_dict = {}, {}, {}, {}
        imgs_paths_dict = {}

        # loop over all scene
        for cur_scene in scene_list:
            # follow MVSNeRF, test use 'val' split, train use 'train' split
            train_views = pairs_dict[f'{cur_scene}_train']
            test_views = pairs_dict[f'{cur_scene}_val']

            cur_scene_info = self.build_test_metas_per_scene(cur_scene, train_views, test_views, method)
            metas.extend(cur_scene_info[0])
            intrinsics_dict.update(cur_scene_info[1])
            world2cams_dict.update(cur_scene_info[2])
            cam2worlds_dict.update(cur_scene_info[3])
            near_fars_dict.update(cur_scene_info[4])
            imgs_paths_dict.update(cur_scene_info[5])

        return metas, intrinsics_dict, world2cams_dict, cam2worlds_dict, near_fars_dict, imgs_paths_dict

    def build_test_metas_per_scene(self, scene_name, train_views=None, test_views=None, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []
        if self.eval_mode == 'mvsnerf':
            if train_views is None or test_views is None:
                raise Exception('Must provide train and test views ids for mvsnerf evaluation.')

            id_list = [*train_views, *test_views]
            meta_filepath = os.path.join(self.root_dir, scene_name, "transforms_train.json")  # all views come from the raw train split

            intrinsics, world2cams, cam2worlds, \
                near_fars, imgs_paths = self.build_camera_info_per_scene(id_list, meta_filepath, scene_name)
        else:
            # process train views
            train_names = [x for x in os.listdir(os.path.join(self.root_dir, scene_name, 'train')) if x.endswith('png')]
            train_views = sorted(list(set([int(x.split('.')[0].split('_')[-1]) for x in train_names])))
            train_views = [f'train_{x}' for x in train_views]
            meta_filepath = os.path.join(self.root_dir, scene_name, "transforms_train.json")
            intrinsics, world2cams, cam2worlds, \
                near_fars, imgs_paths = self.build_camera_info_per_scene(train_views, meta_filepath, scene_name)
            # process test views
            test_names = [x for x in os.listdir(os.path.join(self.root_dir, scene_name, 'test')) if x.endswith('png')]
            test_views = sorted(list(set([int(x.split('.')[0].split('_')[-1]) for x in test_names])))
            test_views = [f'test_{x}' for x in test_views]
            meta_filepath = os.path.join(self.root_dir, scene_name, "transforms_test.json")
            test_data = self.build_camera_info_per_scene(test_views, meta_filepath, scene_name)
            intrinsics.update(test_data[0])
            world2cams.update(test_data[1])
            cam2worlds.update(test_data[2])
            near_fars.update(test_data[3])
            imgs_paths.update(test_data[4])

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

            metas.append((scene_name, target_view, src_idx))

        return metas, intrinsics, world2cams, cam2worlds, near_fars, imgs_paths

    def build_camera_info_per_scene(self, id_list, meta_filepath, scene_name):
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        with open(meta_filepath, 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        focal = 0.5 * 800. / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal = focal * w / 800.  # modify focal length to match size self.img_wh
        intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])  # shared by the whole dataset
        near_far = [2.0, 6.0]  # shared by the whole dataset

        intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
        imgs_paths = {}
        for view_idx in id_list:
            intrinsics[f'{scene_name}_{view_idx}'] = intrinsic
            near_fars[f'{scene_name}_{view_idx}'] = near_far

            if self.eval_mode == 'mvsnerf':
                frame = meta['frames'][view_idx]
            else:
                frame = meta['frames'][int(view_idx.split('_')[-1])]

            c2w = np.array(frame['transform_matrix']) @ blender2opencv
            cam2worlds[f'{scene_name}_{view_idx}'] = c2w
            w2c = np.linalg.inv(c2w)
            world2cams[f'{scene_name}_{view_idx}'] = w2c
            imgs_paths[f'{scene_name}_{view_idx}'] = f"{frame['file_path']}.png"

        return intrinsics, world2cams, cam2worlds, near_fars, imgs_paths

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}

        scene, target_view, src_views = self.metas[idx]
        view_ids = [src_views[i] for i in range(self.n_views)] + [target_view]

        imgs, intrinsics, w2cs, near_fars = [], [], [], []
        img_wh = np.array(self.img_wh).astype('int')
        for vid in view_ids:
            img_filename = os.path.join(self.root_dir, scene, self.imgs_paths_dict[f'{scene}_{vid}'])
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
        sample['near_fars'] = np.stack(near_fars).astype(np.float32)

        sample['scene'] = scene
        sample['img_wh'] = img_wh

        if isinstance(view_ids[0], str):
            view_ids = [int(x.split('_')[-1]) for x in view_ids]
        sample['view_ids'] = np.array(view_ids)

        return sample
