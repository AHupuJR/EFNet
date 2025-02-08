from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate


class VoxelnpzPngSingleDeblurDataset(data.Dataset):
    """Paired vxoel(npz) and blurry image (png) dataset for event-based single image deblurring.
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----voxel
    |    |    |----SEQNAME_%5d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...

    
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VoxelnpzPngSingleDeblurDataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.dataroot_voxel = Path(opt['dataroot_voxel'])
        self.split = 'train' if opt['phase'] == 'train' else 'val'  # train or val
        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'blur'), suffix='.png'))
        blur_frames = [os.path.join(self.dataroot, 'blur', blur_frame) for blur_frame in blur_frames]
        
        sharp_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'sharp'), suffix='.png'))
        sharp_frames = [os.path.join(self.dataroot, 'sharp', sharp_frame) for sharp_frame in sharp_frames]

        event_frames = sorted(recursive_glob(rootdir=self.dataroot_voxel, suffix='.npz'))
        event_frames = [os.path.join(self.dataroot_voxel, event_frame) for event_frame in event_frames]
        
        assert len(blur_frames) == len(sharp_frames) == len(event_frames), f"Mismatch in blur ({len(blur_frames)}), sharp ({len(sharp_frames)}), and event ({len(event_frames)}) frame counts."

        for i in range(len(blur_frames)):
            self.dataPath.append({
                'blur_path': blur_frames[i],
                'sharp_path': sharp_frames[i],
                'event_paths': event_frames[i],
            })
        logger = get_root_logger()
        logger.info(f"Dataset initialized with {len(self.dataPath)} samples.")

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # import pdb; pdb.set_trace()

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        image_path = self.dataPath[index]['blur_path']
        gt_path = self.dataPath[index]['sharp_path']
        event_path = self.dataPath[index]['event_paths']

        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)
        # get GT
        img_bytes = self.file_client.get(gt_path)    # 'gt'
        img_gt = imfrombytes(img_bytes, float32=True)

        voxel = np.load(event_path)['voxel']

        ## Data augmentation
        # voxel shape: h,w,c
        # crop
        if gt_size is not None:
            img_gt, img_lq, voxel = triple_random_crop(img_gt, img_lq, voxel, gt_size, scale, gt_path)

        # flip, rotate
        total_input = [img_lq, img_gt, voxel] 
        img_results = augment(total_input, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results) # hwc -> chw
        img_lq, img_gt, voxel = img_results

        ## Norm voxel
        if self.norm_voxel:
            voxel = voxel_norm(voxel)

        origin_index = os.path.basename(image_path).split('.')[0]

        return {'frame': img_lq, 'frame_gt': img_gt, 'voxel': voxel, 'image_name': origin_index}

    def __len__(self):
        return len(self.dataPath)
