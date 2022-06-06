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


def get_mask_list():
    mask_list = []
    for sub_dir in mask_dir.iterdir():
        mask_list.extend(sorted((sub_dir / '3. Hard Exudates').glob('*tif')))
    return mask_list


if __name__ == "__main__":
    dataset_dir = Path("/homec/kuanghl/Dataset/IDRiD Segmentation/1. Original Images/")
    mask_dir = Path("/homec/kuanghl/Dataset/IDRiD Segmentation/2. All Segmentation Groundtruths")
    # lesion_list = ['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', '4. Soft Exudates']
    mask_list = get_mask_list()

    output_folder = Path("/homec/kuanghl/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task022_IDRID")
    img_dir = output_folder / "imagesTr"
    lab_dir = output_folder / "labelsTr"
    img_dir_te = output_folder / "imagesTs"
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    all_ids = []

    for img_path in dataset_dir.rglob('*.jpg'):
        all_ids.append(img_path.stem)
        img = Image.open(img_path)
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1], 1), order=3)
        convert_2d_image_to_nifti(arr, (img_dir / img_path.stem).as_posix(), is_seg=False)

    for seg_path in mask_list:
        img = Image.open(seg_path).convert('L')
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1]), order=1)
        p = (lab_dir / seg_path.stem[:8]).as_posix()
        convert_2d_image_to_nifti(arr, p, is_seg=True, transform=lambda x: (x > 0).astype(np.uint8))

    json_dict = OrderedDict()
    json_dict['name'] = "IDRID"
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
    test_ids = [i.stem for i in Path("/homec/kuanghl/Dataset/IDRiD Segmentation/1. Original Images/b. Testing Set/").glob('*.jpg')]
    train_ids = list(set(all_ids) - set(test_ids))
    assert len(train_ids) + len(test_ids) == len(all_ids)
    val_ids = test_ids
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = train_ids
    splits[-1]['val'] = test_ids
    splits[-1]['test'] = test_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
