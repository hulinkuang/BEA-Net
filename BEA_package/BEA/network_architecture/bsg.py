import torch
import kornia as K
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.initialization import InitWeights_He


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


def Norm_layer(norm_cfg, inplanes):
    norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
    if norm_cfg == 'BN':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm2d(inplanes, **norm_op_kwargs)

    return out


class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1,
                 norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg)

    def forward(self, x):
        x = self.nonlin(self.norm(self.conv(x)))
        return x


class PoolingConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(PoolingConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.nonlin(self.norm(self.conv(x)))
        return self.max_pool(x)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(MultiScaleConv, self).__init__()
        self.conv1 = ConvNormNonlin(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0,
                                    norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.conv2 = ConvNormNonlin(in_channels, in_channels // 4, norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.conv3 = ConvNormNonlin(in_channels, in_channels // 4, norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.nonlin = Activation_layer(activation_cfg)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.out_conv(x)
        x = self.nonlin(x)

        return x


class Body(nn.Module):
    def __init__(self, channels, layer, norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(Body, self).__init__()
        self.down = ConvNormNonlin(channels, channels, kernel_size=3, stride=2, padding=1,
                                   norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layer = layer
        if layer > 3:
            self.flow = nn.Conv2d(channels * 2, channels, kernel_size=1)
            self.norm = Norm_layer(norm_cfg, channels)
        else:
            self.down1 = ConvNormNonlin(channels, channels, kernel_size=3, stride=2, padding=1,
                                        norm_cfg=norm_cfg, activation_cfg=activation_cfg)
            self.flow = nn.Conv2d(channels * 3, channels, kernel_size=1)
            self.norm = Norm_layer(norm_cfg, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        down = self.down(x)
        if self.layer > 3:
            seg_down = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=True)
            flow = torch.cat([x, seg_down], dim=1)
            flow = self.norm(self.flow(flow))
        else:
            down1 = self.down1(x)
            seg_down = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=True)
            seg_down1 = F.interpolate(down1, size=(H, W), mode='bilinear', align_corners=True)
            flow = torch.cat([x, seg_down, seg_down1], dim=1)
            flow = self.norm(self.flow(flow))
        attn = torch.sigmoid(flow)
        return attn


class Edge(nn.Module):
    def __init__(self, channels, layer, norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(Edge, self).__init__()
        if layer == 1:
            self.conv = ConvNormNonlin(channels * 3, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        elif layer == 2:
            self.conv = ConvNormNonlin(channels * 2, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        else:
            self.conv = ConvNormNonlin(channels, channels, kernel_size=1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg)
        self.layer = layer

    @staticmethod
    def sobel_edge(image):
        B, C, H, W = image.shape
        kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        kernels = torch.tensor(kernels)[:, None, None].repeat(1, C, 1, 1, 1).type_as(image).to(image.device)
        sobel_x = F.conv2d(image, kernels[0], stride=1, padding=1, groups=C)
        sobel_y = F.conv2d(image, kernels[1], stride=1, padding=1, groups=C)

        return sobel_x, sobel_y

    def forward(self, x):
        seg_edge0 = K.filters.sobel(x)

        x1 = F.max_pool2d(x, kernel_size=2)
        seg_edge1 = K.filters.sobel(x1)
        seg_edge1 = F.interpolate(seg_edge1, scale_factor=(2, 2), mode='bilinear')

        x2 = F.max_pool2d(x, kernel_size=4)
        seg_edge2 = K.filters.sobel(x2)
        seg_edge2 = F.interpolate(seg_edge2, scale_factor=(4, 4), mode='bilinear')

        seg_edge = seg_edge0
        if self.layer == 1:
            seg_edge = torch.cat([seg_edge0, seg_edge1, seg_edge2], dim=1)
        elif self.layer == 2:
            seg_edge = torch.cat([seg_edge0, seg_edge1], dim=1)

        seg_edge = self.conv(seg_edge)
        return seg_edge
    
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')

    def forward(self, x):
        x = self.conv(self.up(x))
        x = torch.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_num_features=32, n_layer=5, convolutional_upsampling=False,
                 norm_cfg='IN', activation_cfg='LeakyReLU'):
        super(UNet, self).__init__()

        encoder = []
        in_features = in_channels
        out_features = base_num_features
        for i in range(n_layer-1):
            if i == 0:
                encoder.append(
                    nn.Sequential(
                        ConvNormNonlin(in_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        ConvNormNonlin(out_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg)
                    )
                )
            elif i < 2:
                encoder.append(
                    nn.Sequential(
                        # PoolingConv(in_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        ConvNormNonlin(in_features, out_features, stride=2, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        ConvNormNonlin(out_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                    )
                )
            else:
                encoder.append(
                    nn.Sequential(
                        PoolingConv(in_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        MultiScaleConv(in_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        MultiScaleConv(out_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                    )
                )
            in_features = out_features
            out_features = int(out_features * 2)

        encoder.append(
            nn.Sequential(
                PoolingConv(in_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                MultiScaleConv(in_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                MultiScaleConv(out_features, out_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
            )
        )

        decoder = []
        up = []
        up_edge = []
        de = []
        for i in range(n_layer-1):
            if not convolutional_upsampling:
                up.append(Upsample(out_features))
                up_edge.append(Upsample(out_features))
            else:
                up.append(nn.ConvTranspose2d(out_features, out_features // 2, kernel_size=2, stride=2))
                up_edge.append(nn.ConvTranspose2d(out_features, out_features // 2, kernel_size=2, stride=2))
            if i < 2:
                decoder.append(
                    nn.Sequential(
                        MultiScaleConv(out_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        MultiScaleConv(in_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg)
                    )
                )
                de.append(MultiScaleConv(out_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg))
            else:
                decoder.append(
                    nn.Sequential(
                        ConvNormNonlin(out_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg),
                        ConvNormNonlin(in_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg)
                    )
                )
                de.append((ConvNormNonlin(out_features, in_features, norm_cfg=norm_cfg, activation_cfg=activation_cfg)))
            out_features = in_features
            in_features = in_features // 2

        body = []
        edge = []
        for i in range(n_layer):
            body.append(Body(channels=base_num_features * 2 ** i, layer=i + 1,
                             norm_cfg=norm_cfg, activation_cfg=activation_cfg))
            edge.append(Edge(channels=base_num_features * 2 ** i, layer=i + 1,
                             norm_cfg=norm_cfg, activation_cfg=activation_cfg))

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.up = nn.ModuleList(up)
        self.up_edge = nn.ModuleList(up_edge)
        self.de = nn.ModuleList(de)
        self.body = nn.ModuleList(body)
        self.edge = nn.ModuleList(edge)
        self.n_layer = n_layer

        self.final_body = nn.Sequential(
            nn.Conv2d(base_num_features, 3, kernel_size=3, stride=1, padding=1),
            Activation_layer(activation_cfg),
            nn.Conv2d(3, num_classes, kernel_size=1)
        )
        self.final_edge = nn.Sequential(
            nn.Conv2d(base_num_features, 3, kernel_size=3, stride=1, padding=1),
            Activation_layer(activation_cfg),
            nn.Conv2d(3, num_classes, kernel_size=1)
        )
        self.final_seg = nn.Sequential(
            nn.Conv2d(base_num_features, 3, kernel_size=3, stride=1, padding=1),
            Activation_layer(activation_cfg),
            nn.Conv2d(3, num_classes, kernel_size=1)
        )
        self.apply(InitWeights_He())

    def forward(self, x):
        features = []

        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        body = self.body[-1](features[-1])
        d_edge = self.edge[-1](features[-1])
        for i in range(self.n_layer-1):
            x = self.up[i](body)
            x = torch.cat([features[-(i + 2)], x], dim=1)
            x = self.decoder[i](x)
            body = self.body[-(i+2)](x)
            edge = self.edge[-(i+2)](x)
            d_edge = self.up_edge[i](d_edge)
            d_edge = torch.cat([edge, d_edge], dim=1)
            d_edge = self.de[i](d_edge)

        body_out = self.final_body(body)
        edge_out = self.final_edge(d_edge)
        seg_out = self.final_seg(body + d_edge)

        return seg_out, body_out, edge_out


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256).cuda()
    model = UNet(n_layer=5, convolutional_upsampling=True).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print(total / 1e6)
    outs = model(x)
    print(outs[0].shape)
