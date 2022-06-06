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

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads


def get_subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = []
    for home, _, files in os.walk(folder):
        for filename in files:
            if os.path.isfile(os.path.join(home, filename)) \
                    and (prefix is None or filename.startswith(prefix)) \
                    and (suffix is None or filename.endswith(suffix)):
                res.append(l(home, filename))

    if sort:
        res.sort()
    return res


def zscore_norm(image):
    """ Normalise the image intensity by the mean and standard deviation """
    val_l = 0  # 像素下限
    val_h = 60
    roi = np.where((image >= val_l) & (image <= val_h))
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    image2 = np.copy(image).astype(np.float32)
    image2[image < val_l] = val_l  # val_l
    image2[image > val_h] = val_h

    eps = 1e-6
    image2 = (image2 - mu) / (sigma + eps)
    return image2


def convert_2d_image_to_nifti(img: np.ndarray, output_name: str, spacing=(999, 1, 1), transform=None,
                              is_seg: bool = False) -> None:
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
    train_dir = "/homec/kuanghl/Dataset/AIS_split/train"
    val_dir = "/homec/kuanghl/Dataset/AIS_split/val"
    test_dir = "/homec/kuanghl/Dataset/AIS_split/test"

    data_dir = "/homec/kuanghl/Dataset/AIS"
    nii_files_data = get_subfiles(train_dir, True, "CT", "nii.gz", True)
    file_id = OrderedDict()
    for k, v in zip(nii_files_data, range(1, len(nii_files_data) + 1)):
        file_id[k.split('/')[-2]] = v
    output_folder = "/homec/kuanghl/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task019_AIS2D"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    train_data = get_subfiles(train_dir, True, "CT", "nii.gz", True)
    train_seg = get_subfiles(train_dir, True, "GT_easy", "nii.gz", True)
    val_data = get_subfiles(val_dir, True, "CT", "nii.gz", True)
    val_seg = get_subfiles(val_dir, True, "GT_easy", "nii.gz", True)
    test_data = get_subfiles(test_dir, True, "CT", "nii.gz", True)
    test_seg = get_subfiles(test_dir, True, "GT_easy", "nii.gz", True)

    train_ids = []
    val_ids = []
    test_ids = []

    for img_path, seg_path in zip(train_data, train_seg):
        img_itk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img = zscore_norm(sitk.GetArrayFromImage(img_itk))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        for idx, (img_slice, seg_slice) in enumerate(zip(img, seg)):
            if np.sum(seg_slice) < 100:
                continue
            pat_id = f"AIS{img_path.split('/')[-2]}_slice_{idx}"
            train_ids.append(pat_id)
            convert_2d_image_to_nifti(img_slice, f"{img_dir}/{pat_id}")
            convert_2d_image_to_nifti(seg_slice, f"{lab_dir}/{pat_id}", is_seg=True)
    for img_path, seg_path in zip(val_data, val_seg):
        img_itk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img = zscore_norm(sitk.GetArrayFromImage(img_itk))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        for idx, (img_slice, seg_slice) in enumerate(zip(img, seg)):
            pat_id = f"AIS{img_path.split('/')[-2]}_slice_{idx}"
            val_ids.append(pat_id)
            convert_2d_image_to_nifti(img_slice, f"{img_dir}/{pat_id}")
            convert_2d_image_to_nifti(seg_slice, f"{lab_dir}/{pat_id}", is_seg=True)
    for img_path, seg_path in zip(test_data, test_seg):
        img_itk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img = zscore_norm(sitk.GetArrayFromImage(img_itk))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        for idx, (img_slice, seg_slice) in enumerate(zip(img, seg)):
            pat_id = f"AIS{img_path.split('/')[-2]}_slice_{idx}"
            test_ids.append(pat_id)
            convert_2d_image_to_nifti(img_slice, f"{img_dir}/{pat_id}")
            convert_2d_image_to_nifti(seg_slice, f"{lab_dir}/{pat_id}", is_seg=True)
    all_ids = train_ids + val_ids + test_ids
    json_dict = OrderedDict()
    json_dict['name'] = "AID"
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "0",
        "1": "1"
    }

    json_dict['numTraining'] = len(all_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             all_ids]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    # create a dummy split (patients need to be separated)
    splits = list()
    splits.append(OrderedDict())
    splits[-1]['train'] = train_ids
    splits[-1]['val'] = val_ids
    splits[-1]['test'] = test_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
