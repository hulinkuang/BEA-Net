#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import cv2
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from unittest import TestCase

import SimpleITK as sitk
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.configuration import default_num_threads
from sklearn.model_selection import train_test_split


def resize_seg(itk: sitk.SimpleITK.Image):
    seg = sitk.GetArrayFromImage(itk)
    scale = [1, 0.5, 0.5]
    new_shape = np.array(seg.shape).astype(float)
    new_shape = np.round(new_shape).astype(int)
    for i in range(3):
        new_shape[i] *= scale[i]
    resize_arr = resize_segmentation(seg, new_shape, order=0, cval=0)
    out_itk = sitk.GetImageFromArray(resize_arr)

    out_itk.SetSpacing(itk.GetSpacing())
    out_itk.SetOrigin(itk.GetOrigin())
    out_itk.SetDirection(itk.GetDirection())


def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def convert_2d_image_to_nifti(img: np.ndarray, output_name: str, spacing=(999, 1, 1), transform=None, is_seg: bool = False) -> None:
    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):
        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_name + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_name + ".nii.gz")


if __name__ == "__main__":
    dataset_dir = Path("/home/kuanghl/Dataset/CVC-ClinicDB/")
    data_path = dataset_dir / "Original"
    masks_path = dataset_dir / "Ground Truth"

    output_folder = Path("/home/kuanghl/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task310_CVC_ClinicDB")
    img_dir = output_folder / "imagesTr"
    lab_dir = output_folder / "labelsTr"
    img_dir_te = output_folder / "imagesTs"
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    all_ids = []

    for img_path in data_path.glob('*.tif'):
        all_ids.append(img_path.stem)
        img = cv2.imread(img_path.as_posix())
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1], 1), order=3)
        convert_2d_image_to_nifti(arr, (img_dir / img_path.stem).as_posix(), is_seg=False)

    for seg_path in masks_path.glob('*.tif'):
        img = cv2.imread(seg_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1]), order=1)
        convert_2d_image_to_nifti(arr, (lab_dir / seg_path.stem).as_posix(), is_seg=True,
                                  transform=lambda x: (x > 0).astype(int))

    json_dict = OrderedDict()
    json_dict['name'] = "CVC-ClinicDB"
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "Red",
        "1": "Green",
        "2": "Blue"
    }

    json_dict['labels'] = {
        "0": "0",
        "1": "1"
    }

    json_dict['numTraining'] = len(all_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in all_ids]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    # create a dummy split (patients need to be separated)
    splits = []
    splits.append(OrderedDict())
    train, test = train_test_split(all_ids, test_size=0.201, shuffle=True, random_state=99)
    train, val = train_test_split(train, test_size=0.125, shuffle=True, random_state=99)
    splits[-1]['train'] = train
    splits[-1]['val'] = val
    splits[-1]['test'] = test
    splits[-1]['all'] = all_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
