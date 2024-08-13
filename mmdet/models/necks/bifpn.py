import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16
from ..builder import NECKS


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        try:
            nn.init.constant_(module.bias, bias)
        except AttributeError:
            pass


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SepConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SepConvBNReLU, self).__init__()

        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


@NECKS.register_module()
class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False, ):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = SepConvBNReLU(
                in_channels[i],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level - self.start_level))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvBNReLU(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.1,
                 eps=0.0001):
        super(BiFPNModule, self).__init__()
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        for jj in range(2):
            for i in range(self.levels - 1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    SepConvBNReLU(
                        channels,
                        channels,
                        3,
                        stride=1,
                        padding=1)
                )
                self.bifpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps  # normalize
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (w1[0, i - 1] * pathtd[i - 1] + w1[1, i - 1] * F.interpolate(
                pathtd[i], scale_factor=2, mode='nearest')) / (w1[0, i - 1] + w1[1, i - 1] + self.eps)
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (w2[0, i] * pathtd[i + 1] + w2[1, i] * F.max_pool2d(pathtd[i], kernel_size=2) +
                             w2[2, i] * inputs_clone[i + 1]) / (w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1

        pathtd[levels - 1] = (w1[0, levels - 1] * pathtd[levels - 1] + w1[1, levels - 1] * F.max_pool2d(
            pathtd[levels - 2], kernel_size=2)) / (w1[0, levels - 1] + w1[1, levels - 1] + self.eps)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd