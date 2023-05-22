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


import torch
from torch import einsum
from torch import nn
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper
from monai.networks.utils import one_hot
import kornia as K


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l
    

class NewLoss(nn.Module):
    def __init__(self, w1=1, w2=0.1, w3=0.08, loss=None):
        super(NewLoss, self).__init__()
        self.dc = GDiceLoss(softmax_helper)
        self.ce = WeightedCrossEntropyLoss()
        self.loss = loss

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        y = y[0].squeeze(2)
        x = [i.squeeze(2) for i in x]
        y_edge = K.filters.canny(y)[1]
        y_body = (y - y_edge) > 0
        seg_loss = self.loss(x[0], y)
        body_loss = self.ce(x[1], y_body)
        edge_loss = self.ce(x[-1], y_edge)

        return self.w1 * seg_loss + self.w2 * body_loss + self.w3 * edge_loss


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-6):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y)
        shp_y = gt.shape  # (batch size,1,x,y)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = - 2 * (einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc + 1


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        y_onehot = one_hot(target, num_classes=2, dim=1)
        weight = self._class_weights(y_onehot)
        return F.cross_entropy(input_, y_onehot, weight=weight, reduction='mean')

    @staticmethod
    def _class_weights(input_):
        # normalize the input_ first
        flattened = flatten(input_)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = (nominator / denominator).detach()
        return class_weights


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
