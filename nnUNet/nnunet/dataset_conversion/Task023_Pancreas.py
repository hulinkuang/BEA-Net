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
from pathlib import Path
from scipy.ndimage import zoom
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


def convert_dicom(path: Path):
    itk = sitk.ReadImage(path.as_posix())
    arr = sitk.GetArrayFromImage(itk)
    arr = zoom(np.squeeze(arr), zoom=(0.5, 0.5), order=3)
    pat_id = f"{path.parts[-4]}_{path.stem[2:]}"
    out_name = (img_dir / pat_id).as_posix()
    convert_2d_image_to_nifti(arr, out_name, is_seg=False)
    return pat_id

def convert_nifti(path: Path):
    itk = sitk.ReadImage(path.as_posix())
    arr = sitk.GetArrayFromImage(itk)
    for i in range(arr.shape[0]):
        a = zoom(arr[i], zoom=(0.5, 0.5), order=1)
        out_name = f"PANCREAS_{path.name[5:-7]}_{str(i+1).zfill(3)}"
        out_name = (lab_dir / out_name).as_posix()
        convert_2d_image_to_nifti(a, out_name, is_seg=True)

if __name__ == "__main__":
    train_dir = Path("/homec/kuanghl/Dataset/Pancreas/Train/")
    test_dir = Path("/homec/kuanghl/Dataset/Pancreas/Test/")

    output_folder = Path("/homec/kuanghl/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task023_Pancreas")
    img_dir = output_folder / "imagesTr"
    lab_dir = output_folder / "labelsTr"
    img_dir_te = output_folder / "imagesTs"
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    train_data = sorted(train_dir.rglob('*.dcm'))
    train_seg = sorted(train_dir.rglob('*.nii.gz'))

    test_data = sorted(test_dir.rglob('*.dcm'))
    test_seg = sorted(test_dir.rglob('*.nii.gz'))

    p = Pool(default_num_threads)
    train_ids = p.map(convert_dicom, train_data)
    p.map(convert_nifti, train_seg)
    test_ids = p.map(convert_dicom, test_data)
    p.map(convert_nifti, test_seg)
    p.close()
    p.join()
    
    all_ids = train_ids + test_ids
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
    splits[-1]['val'] = test_ids
    splits[-1]['test'] = test_ids

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
