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
    dataset_dir = "/homec/kuanghl/Dataset/ISIC2018data"
    train_data_path = os.path.join(dataset_dir, 'data_train.npy')
    val_data_path = os.path.join(dataset_dir, 'data_val.npy')
    test_data_path = os.path.join(dataset_dir, 'data_test.npy')

    train_data = np.load(train_data_path)
    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    train_seg_data = np.load("/homec/kuanghl/Dataset/ISIC2018data/mask_train.npy")
    val_seg_data = np.load("/homec/kuanghl/Dataset/ISIC2018data/mask_val.npy")
    test_seg_data = np.load("/homec/kuanghl/Dataset/ISIC2018data/mask_test.npy")

    output_folder = "/homec/kuanghl/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task018_ISIC"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    train_ids = []
    test_ids = []
    val_ids = []

    for idx, img_tr in enumerate(train_data):
        train_ids.append(f"train_{idx+1}")
        convert_2d_image_to_nifti(img_tr, f"{img_dir}/train_{idx+1}", is_seg=False)

    for idx, img_val in enumerate(val_data):
        val_ids.append(f"val_{idx+1}")
        convert_2d_image_to_nifti(img_val, f"{img_dir}/val_{idx+1}", is_seg=False)

    for idx, img_ts in enumerate(test_data):
        test_ids.append(f"test_{idx+1}")
        convert_2d_image_to_nifti(img_ts, f"{img_dir}/test_{idx+1}", is_seg=False)

    for idx, img_tr in enumerate(train_seg_data):
        convert_2d_image_to_nifti(img_tr, f"{lab_dir}/train_{idx+1}", is_seg=True,
                                  transform=lambda x: (x > 0).astype(int))

    for idx, img_val in enumerate(val_seg_data):
        convert_2d_image_to_nifti(img_val, f"{lab_dir}/val_{idx + 1}", is_seg=True,
                                  transform=lambda x: (x > 0).astype(int))

    for idx, img_test in enumerate(test_seg_data):
        convert_2d_image_to_nifti(img_test, f"{lab_dir}/test_{idx + 1}", is_seg=True,
                                  transform=lambda x: (x > 0).astype(int))

    all_ids = train_ids + val_ids + test_ids
    json_dict = OrderedDict()
    json_dict['name'] = "ISIC"
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
    splits[-1]['train'] = train_ids
    splits[-1]['val'] = val_ids
    splits[-1]['test'] = test_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
