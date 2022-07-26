# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from .raw_event_dataset import *
from .noise_function import add_noise_to_voxel, put_hot_pixels_in_voxel_
import argparse
import h5py
import torch
import time


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="/scratch/leisun/Datasets/GOPRO_fullsize_h5/test", help="Path to hdf5 file")
parser.add_argument("--save_path", default="/scratch/e_work/GOPRO_SCER/test")
parser.add_argument("--voxel_method", default="SCER_esim", help="SCER_esim, SCER_real_data")
parser.add_argument("--add_noise", default=True, help="add noisy to voxel like hot pixel")

exposure_time = 1/240
num_bins = 6
num_pixels = 1280*720

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

def voxel2mask(voxel):
    mask_final = np.zeros_like(voxel[0, :, :])
    mask = (voxel != 0)
    for i in range(mask.shape[0]):
        mask_final = np.logical_or(mask_final, mask[i, :, :])
    # to uint8 image
    mask_img = mask_final * np.ones_like(mask_final) * 255
    mask_img = mask_img[..., np.newaxis] # H,W,C
    mask_img = np.uint8(mask_img)

    return mask_img

def main():
    # dataset settings
    # voxel_method = {'method': 'SCER_esim'}
    voxel_method = {'method': args.voxel_method}

    has_exposure_time = False # false if esim, true if our seems data
    # no data augmentation
    data_augment = {}
    dataset_kwargs = {'transforms': data_augment, 'voxel_method': voxel_method, 'num_bins': num_bins,
                      'has_exposure_time': has_exposure_time, 'combined_voxel_channels': True, 'voxel_temporal_bilinear': True} # voxel_temporal_bilinear for voxel grid

    file_folder_path = args.input_path
    output_file_folder_path = args.save_path
    h5_file_paths = [os.path.join(file_folder_path, s) for s in os.listdir(file_folder_path)]
    output_h5_file_paths = [os.path.join(output_file_folder_path, s) for s in os.listdir(file_folder_path)]
    # for each h5_file
    for i in range(len(h5_file_paths)):
        print("processing file: {}".format(h5_file_paths[i]))
        # h5_file = h5py.File(h5_file_paths[i], 'a')
        h5_file = h5py.File(output_h5_file_paths[i],'a')
        # voxel_dset = h5_file.create_group("voxels")
        # N = int(os.path.basename(h5_file_paths[i]).split('_')[1]) # num of the sharp img for blur image produce
        dataset_kwargs.update({'data_path':h5_file_paths[i]})
        dloader = GoproEsimH5Dataset(**dataset_kwargs)
        num_img = 0
        for item in dloader:
            voxel=item['voxel']
            num_events = item['num_events']
            blur = item['frame']  # C,H,W
            sharp = item['frame_gt']

            # add noise to voxel Here
            if args.add_noise:
                # print("Add noisy to voxels")
                voxel = add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.05)
                put_hot_pixels_in_voxel_(voxel, hot_pixel_range=20, hot_pixel_fraction=0.00002)

            voxel_np = voxel.numpy() # shape: bin+1,H,W
            blur_np=np.uint8(np.clip(255*blur.numpy(), 0, 255))
            sharp_np=np.uint8(np.clip(255*sharp.numpy(), 0, 255))
            mask_img = voxel2mask(voxel_np)
            #close filter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_img_close = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
            # print(mask_img_close.shape)
            mask_img_close = mask_img_close[np.newaxis,...] # H,W -> C,H,W  C=1

            # save to h5 image
            voxel_dset=h5_file.create_dataset("voxels/voxel{:09d}".format(num_img), data=voxel_np, dtype=np.dtype(np.float32))
            image_dset=h5_file.create_dataset("images/image{:09d}".format(num_img), data=blur_np, dtype=np.dtype(np.uint8))
            sharp_image_dset=h5_file.create_dataset("sharp_images/image{:09d}".format(num_img), data=sharp_np, dtype=np.dtype(np.uint8))
            mask_dset=h5_file.create_dataset("masks/mask{:09d}".format(num_img), data=mask_img_close, dtype=np.dtype(np.uint8))



            voxel_dset.attrs['size']=voxel_np.shape
            image_dset.attrs['size']=blur_np.shape
            sharp_image_dset.attrs['size']=sharp_np.shape
            mask_dset.attrs['size']=mask_img_close.shape


            num_img+=1
        sensor_resolution = [blur_np.shape[1], blur_np.shape[2]]
        h5_file.attrs['sensor_resolution'] = sensor_resolution




if __name__ == "__main__":
    main()

