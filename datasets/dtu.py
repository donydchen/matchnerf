from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import cv2

from misc.utils import read_pfm


class MVSDatasetDTU(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 test_views_method='nearest', n_add_train_views=2, **kwargs):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        assert split in ['train', 'val', 'test'], 'split must be either "train", "val" or "test"!'
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.val_light_idx = 3
        self.val_view_idx = 24
        self.n_add_train_views = n_add_train_views
        self.permute_train_src = True

        self.transform = self.define_transforms()

        if split in ['train', 'val']:
            scene_list_filepath = os.path.join('configs', 'dtu_meta', 'train_all.txt')
            view_pairs_filepath = os.path.join('configs', 'dtu_meta', 'view_pairs.txt')
            self.metas, id_list = self.build_train_metas(scene_list_filepath, view_pairs_filepath)
            self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                self.build_camera_info(id_list)
        else:  # test cases
            scene_list_filepath = os.path.join('configs', 'dtu_meta', 'val_all.txt')
            view_pairs_filepath = os.path.join('configs', 'pairs.th')
            view_pairs = torch.load(view_pairs_filepath)
            train_views, test_views = view_pairs['dtu_train'], view_pairs['dtu_test']
            id_list = [*train_views, *test_views]
            self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                self.build_camera_info(id_list)
            self.metas = self.build_test_metas(scene_list_filepath, train_views, test_views, method=test_views_method)

    def get_name(self):
        dataname = 'dtu'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor(),])
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    def build_train_metas(self, scene_list_filepath, view_pairs_filepath):
        '''Build train metas, get input source views based on the order pre-defined in `view_pairs_filepath`.'''
        metas = []
        # read scene list
        with open(scene_list_filepath) as f:
            scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [self.val_light_idx] if 'train' != self.split else range(7)

        id_list = []
        for scan in scans:
            with open(view_pairs_filepath) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        if self.split == 'val' and ref_view != self.val_view_idx:
                            continue
                        metas += [(scan, light_idx, ref_view, src_views)]
                        id_list.append([ref_view] + src_views)

        id_list = np.unique(id_list)
        return metas, id_list

    def build_camera_info(self, id_list):
        '''Return the camera information for the given id_list'''
        intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
        for vid in id_list:
            proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)

            intrinsic[:2] *= 4
            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics[vid] = intrinsic

            extrinsic[:3, 3] *= self.scale_factor
            world2cams[vid] = extrinsic
            cam2worlds[vid] = np.linalg.inv(extrinsic)

            near_fars[vid] = near_far

        return intrinsics, world2cams, cam2worlds, near_fars

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsic = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsic = extrinsic.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsic = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsic = intrinsic.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        near_far = [depth_min, depth_max]
        return intrinsic, extrinsic, near_far

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample, interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        return depth_h

    def build_test_metas(self, scene_list_filepath, train_views, test_views, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []
        # read scene list
        with open(scene_list_filepath) as f:
            scans = [line.rstrip() for line in f.readlines()]

        light_idx = 3
        for scan in scans:
            for target_view in test_views:
                src_views = self.sorted_test_src_views(target_view, train_views, method)
                metas.append((scan, light_idx, target_view, src_views))

        return metas

    def sorted_test_src_views(self, target_view, train_views, method='nearest'):
        if method == "nearest":
            cam_pos_trains = np.stack([self.cam2worlds_dict[x] for x in train_views])[:, :3, 3]
            cam_pos_target = self.cam2worlds_dict[target_view][:3, 3]
            dis = np.sum(np.abs(cam_pos_trains - cam_pos_target), axis=-1)
            src_idx = np.argsort(dis)
            src_idx = [train_views[x] for x in src_idx]
        elif method == "fixed":
            src_idx = train_views
        else:
            raise Exception('Unknown evaluate method [%s]' % method)
        return src_idx

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, target_view, src_views = self.metas[idx]
        if self.permute_train_src and self.split == 'train':
            ids = torch.sort(torch.randperm(self.n_views + self.n_add_train_views)[:self.n_views])[0]
            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(self.n_views)] + [target_view]
            # print(scan, view_ids)

        # print('view_ids', view_ids, 'light_idx', light_idx, 'scan', scan)

        imgs, intrinsics, w2cs, near_fars = [], [], [], []  # record proj mats between views
        depth = None  # only used for test case
        img_wh = np.round(np.array(self.img_wh) * self.downSample).astype('int')
        for vid in view_ids:
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs.append(img)

            intrinsics.append(self.intrinsics_dict[vid])
            w2cs.append(self.world2cams_dict[vid])
            near_fars.append(self.near_fars_dict[vid])

            # read target view depth for evaluation
            if self.split in ['test', 'val'] and vid == target_view:
                depth_filename = os.path.join(self.root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                assert os.path.exists(depth_filename), "Must provide depth for evaluating purpose."
                depth = self.read_depth(depth_filename)
                depth = depth * self.scale_factor

        sample['images'] = torch.stack(imgs).float()  # (V, H, W, 3)
        sample['extrinsics'] = np.stack(w2cs).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack(intrinsics).astype(np.float32)  # (V, 3, 3)
        sample['near_fars'] = np.stack(near_fars).astype(np.float32)
        sample['view_ids'] = np.array(view_ids)
        # sample['light_id'] = np.array(light_idx)
        sample['scene'] = scan
        sample['img_wh'] = img_wh

        if self.split in ['test', 'val'] and depth is not None:
            sample['depth'] = depth.astype(np.float32)

        return sample
