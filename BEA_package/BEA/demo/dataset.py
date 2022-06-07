import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from functools import partial


def preprocess(image_path, is_mask=False):
    def z_score(image):
        img_list = [(i - np.mean(i)) / (np.std(i) + 1e-8) for i in image]
        return np.stack(img_list, axis=0)

    img = Image.open(image_path)
    array = np.array(img)
    if is_mask:
        array[array > 0] = 1
        array = zoom(array, (256 / array.shape[0], 256 / array.shape[1]), order=1)
    else:
        array = array.transpose(2, 0, 1)
        array = zoom(array, (1, 256 / array.shape[1], 256 / array.shape[2]), order=3)
        array = z_score(array)

    return array


class ISIC(Dataset):
    def __init__(self, data_path):
        img_list = sorted(data_path.glob('*[0-9].jpg'))
        mask_list = sorted(data_path.glob('*n.png'))

        self.images = list(map(preprocess, img_list))
        self.masks = list(map(partial(preprocess, is_mask=True), mask_list))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx][:, None].astype(np.float32)
        mask = self.masks[idx].astype(np.int64)

        return img, mask


if __name__ == '__main__':
    dataset = ISIC(Path('images'))
    arr, s = dataset[0]
    print()

