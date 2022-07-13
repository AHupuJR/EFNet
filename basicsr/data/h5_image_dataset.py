# ------------------------------------------------------------------------
# Modified from (https://github.com/TimoStoff/events_contrast_maximization)
# ------------------------------------------------------------------------
from torch.utils import data as data
import pandas as pd
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from torch.utils.data.dataloader import default_collate
import h5py
# local modules
from basicsr.data.h5_augment import *
from torch.utils.data import ConcatDataset


"""
    Data augmentation functions.
    modified from https://github.com/TimoStoff/events_contrast_maximization

    @InProceedings{Stoffregen19cvpr,
    author = {Stoffregen, Timo and Kleeman, Lindsay},
    title = {Event Cameras, Contrast Maximization and Reward Functions: An Analysis},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    } 
"""


def concatenate_h5_datasets(dataset, opt):
    """
    file_path: path that contains the h5 file
    """
    file_folder_path = opt['dataroot']
    

    if os.path.isdir(file_folder_path):
        h5_file_path = [os.path.join(file_folder_path, s) for s in os.listdir(file_folder_path)]
    elif os.path.isfile(file_folder_path):
        h5_file_path = pd.read_csv(file_folder_path, header=None).values.flatten().tolist()
    else:
        raise Exception('{} must be data_file.txt or base/folder'.format(file_folder_path))
    print('Found {} h5 files in {}'.format(len(h5_file_path), file_folder_path))
    datasets = []
    for h5_file in h5_file_path:
        datasets.append(dataset(opt, h5_file))
    return ConcatDataset(datasets)


class H5ImageDataset(data.Dataset):

    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_gt_frame(self, index):
        """
        Get gt frame at index
        @param index: The index of the gt frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['sharp_images']['image{:09d}'.format(index)][:]

    def get_voxel(self, index):
        """
        Get voxels at index
        @param index The index of the voxels to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['voxels']['voxel{:09d}'.format(index)][:]

    def get_mask(self, index):
        """
        Get event mask at index
        @param index The index of the event mask to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['masks']['mask{:09d}'.format(index)][:]


    def __init__(self, opt, data_path, return_voxel=True, return_frame=True, return_gt_frame=True,
            return_mask=False, norm_voxel=True):

        super(H5ImageDataset, self).__init__()
        self.opt = opt
        self.data_path = data_path
        self.seq_name = os.path.basename(self.data_path)
        self.seq_name = self.seq_name.split('.')[0]
        self.return_format = 'torch'

        self.return_voxel = return_voxel
        self.return_frame = return_frame
        self.return_gt_frame = opt.get('return_gt_frame', return_gt_frame)
        self.return_voxel = opt.get('return_voxel', return_voxel)
        self.return_mask = opt.get('return_mask', return_mask)
        
        self.norm_voxel = norm_voxel # -MAX~MAX -> -1 ~ 1 
        self.h5_file = None
        self.transforms={}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


        if self.opt['norm_voxel'] is not None:
            self.norm_voxel = self.opt['norm_voxel']   # -MAX~MAX -> -1 ~ 1 
        
        if self.opt['return_voxel'] is not None:
            self.return_voxel = self.opt['return_voxel']

        if self.opt['crop_size'] is not None:
            self.transforms["RandomCrop"] = {"size": self.opt['crop_size']}
        
        if self.opt['use_flip']:
            self.transforms["RandomFlip"] = {}

        if 'LegacyNorm' in self.transforms.keys() and 'RobustNorm' in self.transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in self.transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]
                del (self.transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)

        if not self.normalize_voxels:
            self.vox_transform = self.transform

        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file['images'].keys())


    def __getitem__(self, index, seed=None):

        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        item={}
        frame = self.get_frame(index)
        if self.return_gt_frame:
            frame_gt = self.get_gt_frame(index)
            frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=False)

        voxel = self.get_voxel(index)
        frame = self.transform_frame(frame, seed, transpose_to_CHW=False)  # to tensor

        # normalize RGB
        if self.mean is not None or self.std is not None:
            normalize(frame, self.mean, self.std, inplace=True)
            if self.return_gt_frame:
                normalize(frame_gt, self.mean, self.std, inplace=True)

        if self.return_frame:
            item['frame'] = frame
        if self.return_gt_frame:
            item['frame_gt'] = frame_gt
        if self.return_voxel:
            item['voxel'] = self.transform_voxel(voxel, seed, transpose_to_CHW=False)
        if self.return_mask:
            mask = self.get_mask(index)
            item['mask'] = self.transform_frame(mask, seed, transpose_to_CHW=False)
            
        item['seq'] = self.seq_name


        return item


    def __len__(self):
        return self.dataset_len

    def transform_frame(self, frame, seed, transpose_to_CHW=False):
        """
        Augment frame and turn into tensor
        @param frame Input frame
        @param seed  Seed for random number generation
        @returns Augmented frame
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255  # H,W,C -> C,H,W

            else:
                frame = torch.from_numpy(frame).float() / 255 # 0-1
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed, transpose_to_CHW):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()# H,W,C -> C,H,W

            else:
                if self.norm_voxel:
                    voxel = torch.from_numpy(voxel).float() / abs(max(voxel.min(), voxel.max(), key=abs))  # -1 ~ 1
                else:
                    voxel = torch.from_numpy(voxel).float()

            if self.vox_transform:
                random.seed(seed)
                voxel = self.vox_transform(voxel)
        return voxel


    @staticmethod
    def collate_fn(data, event_keys=['events'], idx_keys=['events_batch_indices']):
        """
        Custom collate function for pyTorch batching to allow batching events
        """
        collated_events = {}
        events_arr = []
        end_idx = 0
        batch_end_indices = []
        for idx, item in enumerate(data):
            for k, v in item.items():
                if not k in collated_events.keys():
                    collated_events[k] = []
                if k in event_keys:
                    end_idx += v.shape[0]
                    events_arr.append(v)
                    batch_end_indices.append(end_idx)
                else:
                    collated_events[k].append(v)
        for k in collated_events.keys():
            try:
                i = event_keys.index(k)
                events = torch.cat(events_arr, dim=0)
                collated_events[event_keys[i]] = events
                collated_events[idx_keys[i]] = batch_end_indices
            except:
                collated_events[k] = default_collate(collated_events[k])
        return collated_events

