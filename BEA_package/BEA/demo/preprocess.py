import shutil
from collections import OrderedDict
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from PIL import Image
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from scipy.ndimage import zoom


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


output_folder = Path("/homeb/wyh/Codes/BEA-Net/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task026_ISIC")


def preprocess():
    dataset_dir = Path("images")

    img_dir = output_folder / "imagesTr"
    lab_dir = output_folder / "labelsTr"
    img_dir_te = output_folder / "imagesTs"
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)

    all_ids = []

    for img_path in dataset_dir.glob('*[0-9].jpg'):
        all_ids.append(img_path.stem)
        img = Image.open(img_path)
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1], 1), order=3)
        convert_2d_image_to_nifti(arr, (img_dir / img_path.stem).as_posix(), is_seg=False)

    for seg_path in dataset_dir.glob('*n.png'):
        img = Image.open(seg_path).convert('L')
        arr = np.array(img)
        arr = zoom(arr, (256 / arr.shape[0], 256 / arr.shape[1]), order=1)
        convert_2d_image_to_nifti(arr, (lab_dir / seg_path.stem[:-13]).as_posix(), is_seg=True,
                                  transform=lambda x: (x > 0).astype(int))

    json_dict = OrderedDict()
    json_dict['name'] = "ISIC2018"
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
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             all_ids]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    train_ids = all_ids[:-2]
    val_ids = all_ids[-2:-1]
    test_ids = all_ids[-1:]
    splits = [OrderedDict()]
    splits[-1]['train'] = train_ids
    splits[-1]['val'] = val_ids
    splits[-1]['test'] = test_ids

    splits_path = join(output_folder, "splits_final.pkl")
    save_pickle(splits, splits_path)
    task = 'Task026_ISIC'
    os.system(
        '/home/wyh/anaconda3/envs/BSG/bin/python -u /homeb/wyh/Codes/BEA-Net/nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 26 ')
    dst = join(preprocessing_output_dir, task)
    shutil.copy(splits_path, dst)
