# ------------------------------------------------------------------------
# BEA
# ------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from BEA.network_architecture.bsg import UNet
from BEA.network_architecture.neural_network import SegmentationNetwork


class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=None):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.backbone = UNet(in_channels=3, norm_cfg=norm_cfg, activation_cfg=activation_cfg,
                             n_layer=5, convolutional_upsampling=True)

    def forward(self, inputs):
        # # %%%%%%%%%%%%% Body-Edge Segmentation Network
        B, C, D, H, W = inputs.shape
        x = inputs.squeeze(2)
        # x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.backbone(x)
        # x = [F.interpolate(i, size=(H, W), mode='bilinear') for i in x]
        outs = [i.unsqueeze(2) for i in x]

        return outs


class ResTranUnet(SegmentationNetwork):
    """
    ResTran-3D Unet
    """

    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=None, deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, num_classes)  # U_ResTran3D

        self.conv_op = nn.Conv3d
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]
