from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from pathlib import Path
from misc.utils import list_all_images


class MVSDatasetTNT(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 scene_list=None, test_views_method='nearest', eval_mode='mvsnerf', nf_mode='avg', **kwargs):
        assert split in ['test'], 'Only support "test" split for TNT dataset!'

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.max_len = max_len
        self.nf_mode = nf_mode
        self.eval_mode = eval_mode
        if eval_mode == 'gpnr':
            self.test_hold_out = 8  # follow GPNR settings

        self.img_wh = img_wh
        # if img_wh is not None:
        # assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'
        self.transform = self.define_transforms()
        self.scale_factor = 500.

        if scene_list is None:
            scene_list = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        pairs_dict = torch.load(os.path.join('configs', 'pairs.th'))
        self.metas, self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, \
            self.near_fars_dict, self.imgs_paths_dict = self.build_test_metas(scene_list, pairs_dict, method=test_views_method)

    def get_name(self):
        dataname = 'tnt'
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
                train_views = pairs_dict[f'TNT_{cur_scene}_train']
                test_views = pairs_dict[f'TNT_{cur_scene}_val']
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
        # meta_filepath = os.path.join(self.root_dir, scene_name, "poses_bounds.npy")

        intrinsics, world2cams, cam2worlds, \
            near_fars, imgs_paths = self.build_camera_info_per_scene(id_list, None, scene_name)

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
        images_dir = os.path.join(self.root_dir, scene_name, "images")
        cameras_dir = os.path.join(self.root_dir, scene_name, "cams_1")
        intrinsics, world2cams, cam2worlds, near_fars, imgs_paths = {}, {}, {}, {}, {}
        for view_idx in id_list:
            proj_mat_filename = os.path.join(cameras_dir, f'{view_idx:08d}_cam.txt')

            intr, extr, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)

            intrinsics[f'{scene_name}_{view_idx}'] = intr

            extr[:3,3] *= self.scale_factor 
            world2cams[f'{scene_name}_{view_idx}'] = extr

            c2w = np.linalg.inv(extr.astype(np.float32))
            cam2worlds[f'{scene_name}_{view_idx}'] = c2w  # 4x4

            near_fars[f'{scene_name}_{view_idx}'] = np.array([depth_min_*self.scale_factor, depth_max_*self.scale_factor])

            # img_filename = os.path.join(images_dir, f'{view_idx:08d}.jpg')
            imgs_paths[f'{scene_name}_{view_idx}'] = f'{view_idx:08d}.jpg'

        return intrinsics, world2cams, cam2worlds, near_fars, imgs_paths

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        return intrinsics, extrinsics, depth_min, depth_max

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
            ori_w, ori_h = img.size
            img = img.resize(img_wh, Image.LANCZOS)
            img = self.transform(img)
            imgs.append(img)

            raw_intr = self.intrinsics_dict[f'{scene}_{vid}'].copy()
            raw_intr[0] *= img_wh[0] / ori_w
            raw_intr[1] *= img_wh[1] / ori_h
            intrinsics.append(raw_intr)
            w2cs.append(self.world2cams_dict[f'{scene}_{vid}'])
            near_fars.append(self.near_fars_dict[f'{scene}_{vid}'])

        sample['images'] = torch.stack(imgs).float()  # (V, H, W, 3)
        sample['extrinsics'] = np.stack(w2cs).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack(intrinsics).astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['scene'] = scene
        sample['img_wh'] = img_wh

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

        return sample


if __name__ == "__main__":
    # debug and save view pairs: python -m datasets.tnt
    from torch.utils.data import DataLoader
    import yaml

    with open('configs/test_tnt.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Instantiate your dataset
    dataset = MVSDatasetTNT(split='test', **config["data_test"]["tnt"])

    # Optionally, use a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Loop over the whole dataset
    tnt_pairs = {}
    for i, data in enumerate(dataloader):
        scene_name = data["scene"][0]
        if scene_name not in tnt_pairs:
            tnt_pairs[scene_name] = {}
        view_ids = data['view_ids'][0].numpy().tolist()
        src_views = view_ids[:-1]
        tgt_view = view_ids[-1]
        tnt_pairs[scene_name].update({tgt_view: src_views})
        print(scene_name, tgt_view, src_views)

    torch.save(tnt_pairs, "tnt_pairs.th")
