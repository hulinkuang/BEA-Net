# ------------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------------
import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from medpy.metric.binary import dc
from pathlib import Path
from torch.utils.data import DataLoader
from collections import OrderedDict
from BEA.network_architecture.ResTranUnet import ResTranUnet
from dataset import ISIC


def load_checkpoint(checkpoint_path, network):
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    curr_state_dict_keys = list(network.state_dict().keys())
    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in checkpoint['state_dict'].items():
        key = k
        if key not in curr_state_dict_keys and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    network.load_state_dict(new_state_dict)
    return network


def compute_dice(pred_list, label_list):
    dice = 0
    for pred, label in zip(pred_list, label_list):
        dice = dice + dc(pred, label)
    dice = dice / len(pred_list)
    return dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=str, default='0')
    parser.add_argument("-checkpoint", type=str, default='')
    parser.add_argument("-data_path", type=str, default='./images')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = ResTranUnet(norm_cfg='IN', activation_cfg='LeakyReLU')
    if args.checkpoint != '':
        model = load_checkpoint(args.checkpoint, model)
    dataset = ISIC(Path(args.data_path))
    dl = DataLoader(dataset, batch_size=1, shuffle=False)

    prediction_list = []
    mask_list = []
    for img, mask in dl:
        img = img
        output = model(img)
        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
        prediction = prediction.squeeze().detach().numpy()
        mask = mask.squeeze().numpy()
        prediction_list.append(prediction)
        mask_list.append(mask)
    m_dice = compute_dice(prediction_list, mask_list)
    print(f'mean dice: {m_dice}')


if __name__ == "__main__":
    main()
