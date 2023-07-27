# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import ConvModule, xavier_init

from mmseg.models.builder import NECKS


class ConvBlock(nn.Module):
    """
    The ConvBlock module is a basic building block for the decoder.
    """

    def __init__(self, num_channels, momentum=0.9997, eps=4e-5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=num_channels,
            ),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=momentum, eps=eps),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv(input)


class Decode(nn.Module):
    """
    The Decode module is the decoder of the ChangeDetCatBifpn module.
    """

    def __init__(self, num_channels, epsilon=1e-4, momentum=0.9997):
        super(Decode, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = ConvBlock(num_channels, momentum)
        self.conv5_up = ConvBlock(num_channels, momentum)
        self.conv4_up = ConvBlock(num_channels, momentum)
        self.conv3_up = ConvBlock(num_channels, momentum)
        self.conv2_up = ConvBlock(num_channels, momentum)
        self.conv3_down = ConvBlock(num_channels, momentum)
        self.conv4_down = ConvBlock(num_channels, momentum)
        self.conv5_down = ConvBlock(num_channels, momentum)
        self.conv6_down = ConvBlock(num_channels, momentum)
        self.conv7_down = ConvBlock(num_channels, momentum)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p2_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p3_downsample = nn.MaxPool2d(kernel_size=2)
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2))
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(torch.ones(3))
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        Forward function of the Decode module.
        """
        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p2_in, p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_up = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv2_up(weight[0] * p2_in + weight[1] * self.p2_upsample(p3_up))

        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(
            weight[0] * p3_in
            + weight[1] * p3_up
            + weight[2] * self.p3_downsample(p2_out)
        )

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in
            + weight[1] * p4_up
            + weight[2] * self.p4_downsample(p3_out)
        )
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in
            + weight[1] * p5_up
            + weight[2] * self.p5_downsample(p4_out)
        )
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in
            + weight[1] * p6_up
            + weight[2] * self.p6_downsample(p5_out)
        )
        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)
        )

        return p2_out, p3_out, p4_out, p5_out, p6_out, p7_out


@NECKS.register_module()
class ChangeDetCatBifpn(nn.Module):
    """
    The neck of the ChangeDet model with the CatBiFPN structure.

    Args:
        in_channels (tuple[int]): The input channels of the neck.
        num_channels (int): The number of channels in the neck.
        momentum (float): The momentum of the BatchNorm layers. Default: 0.9997.
    """

    def __init__(self, in_channels, num_channels, momentum=0.9997):
        super(ChangeDetCatBifpn, self).__init__()
        self.filters = [_ * 2 for _ in in_channels]
        self.num_channels = num_channels

        self.conv2 = nn.Conv2d(
            self.filters[0], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            self.filters[1], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            self.filters[2], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            self.filters[3], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv6 = nn.Conv2d(
            self.filters[3], self.num_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1
            ),
        )

        self.bifpn = nn.Sequential(
            *[Decode(self.num_channels, momentum=momentum) for _ in range(2)]
        )
        self.out7 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=32, mode="nearest"),
        )
        self.out6 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=16, mode="nearest"),
        )
        self.out5 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=8, mode="nearest"),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="nearest"),
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1)
        )

    def init_weights(self):
        """
        Initialize the weights of the layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        """
        Forward computation of the neck.

        Args:
            x (tuple[Tensor]): The input features.

        Returns:
            tuple[Tensor]: The output features.
        """
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = torch.cat((e2x1, e2x2), 1)
        c3 = torch.cat((e3x1, e3x2), 1)
        c4 = torch.cat((e4x1, e4x2), 1)
        c5 = torch.cat((e5x1, e5x2), 1)

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        return fuse


@NECKS.register_module()
class ChangeDetSubBifpn(nn.Module):
    """
    The neck of the ChangeDet model with the SubBiFPN structure.

    Args:
        in_channels (tuple[int]): The input channels of the neck.
        num_channels (int): The number of channels in the neck.
        momentum (float): The momentum of the BatchNorm layers. Default: 0.9997.
    """

    def __init__(self, in_channels, num_channels, momentum=0.9997):
        super(ChangeDetSubBifpn, self).__init__()
        self.filters = [_ * 1 for _ in in_channels]
        self.num_channels = num_channels

        self.conv2 = nn.Conv2d(
            self.filters[0], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            self.filters[1], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            self.filters[2], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            self.filters[3], self.num_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv6 = nn.Conv2d(
            self.filters[3], self.num_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1
            ),
        )

        self.bifpn = nn.Sequential(
            *[Decode(self.num_channels, momentum=momentum) for _ in range(2)]
        )
        self.out7 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=32, mode="nearest"),
        )
        self.out6 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=16, mode="nearest"),
        )
        self.out5 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=8, mode="nearest"),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="nearest"),
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels // 6, 3, padding=1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        """
        Forward computation of the neck.

        Args:
            x (tuple[Tensor]): The input features.

        Returns:
            tuple[Tensor]: The output features.
        """
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = e2x1 - e2x2
        c3 = e3x1 - e3x2
        c4 = e4x1 - e4x2
        c5 = e5x1 - e5x2

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        # return [p7_out, p6_out, p5_out, p4_out, p3_out, p2_out]
        return fuse


@NECKS.register_module()
class ChangeDetCat(nn.Module):
    def __init__(self):
        super(ChangeDetCat, self).__init__()

    def forward(self, x):
        x1, x2 = x
        out = [torch.cat([x1[ind], x2[ind]], dim=1) for ind in range(len(x1))]
        return out


@NECKS.register_module()
class ChangeDetSubtract(nn.Module):
    def __init__(self):
        super(ChangeDetSubtract, self).__init__()

    def forward(self, x):
        x1, x2 = x
        out = [x1[ind] - x2[ind] for ind in range(len(x1))]
        return out


@NECKS.register_module()
class ChangeDetSubAddCat(nn.Module):
    def __init__(self):
        super(ChangeDetSubAddCat, self).__init__()

    def forward(self, x):
        x1, x2 = x
        out = [
            torch.cat([x1[ind] - x2[ind], x1[ind] + x2[ind]], dim=1)
            for ind in range(len(x1))
        ]
        return out


@NECKS.register_module()
class ChangeDetCatBatch(nn.Module):
    def __init__(self):
        super(ChangeDetCatBatch, self).__init__()

    def forward(self, x):
        x1, x2 = x
        out = [torch.cat([x1[ind], x2[ind]], dim=0) for ind in range(len(x1))]
        return out


class ChangeSELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(
        self,
        channels,
        ratio=16,
        mode="subtract",
        conv_cfg=None,
        act_cfg=(dict(type="ReLU"), dict(type="Sigmoid")),
    ):
        super(ChangeSELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.mode = mode
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if mode == "subtract":
            in_channels = channels
        elif mode == "concat":
            in_channels = channels * 2
        else:
            raise NotImplementedError
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0],
        )
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1],
        )

    def forward(self, x):
        x1, x2 = x.chunk(2)
        if self.mode == "subtract":
            out = torch.abs(x1 - x2)
        else:
            out = torch.cat([x1, x2], dim=1)
        out = self.global_avgpool(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.repeat(2, 1, 1, 1)
        return x * out


@NECKS.register_module()
class ChangeSE(nn.Module):
    def __init__(self, in_channels, se_ratio=16, mode="subtract"):
        super(ChangeSE, self).__init__()
        # in_channels = [_ // 2 for _ in in_channels]
        self.se_layers = nn.ModuleList()
        for in_channel in in_channels:
            self.se_layers.append(ChangeSELayer(in_channel, se_ratio, mode))
        self.in_channels = in_channels

    def forward(self, x):
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        x = [torch.cat([x1[ind], x2[ind]], dim=0) for ind in range(len(x1))]
        x1_out = []
        x2_out = []
        for i in range(len(x)):
            _x1, _x2 = self.se_layers[i](x[i]).chunk(2)
            x1_out.append(_x1)
            x2_out.append(_x2)
        return [x1_out, x2_out]


@NECKS.register_module()
class Warp(nn.Module):
    def __init__(self, in_channels, in_index=0):
        super(Warp, self).__init__()
        assert isinstance(in_index, int)
        self.in_channels = in_channels
        self.in_index = in_index
        self.flow_make = nn.Conv2d(
            in_channels[in_index] * 2, 2, kernel_size=3, padding=1, bias=False
        )

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x):
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        size_ratio = [
            torch.tensor(x1[i].shape[2:]) / torch.tensor(x1[self.in_index].shape[2:])
            for i in range(len(x1))
        ]
        flow = self.flow_make(torch.cat([x1[self.in_index], x2[self.in_index]], dim=1))

        x2_out = []
        for i, _x2 in enumerate(x2):
            size = _x2.shape[2:]
            _flow = F.interpolate(flow, size, mode="bilinear", align_corners=True)
            _flow = _flow * size_ratio[i][:, None, None].type_as(_x2).to(_x2.device)
            warp = self.flow_warp(_x2, _flow, size)
            x2_out.append(warp)
        return [x1, x2_out]


@NECKS.register_module()
class ChangeDetCatBifpnShift(ChangeDetCatBifpn):
    def __init__(
        self,
        in_channels,
        num_channels,
        in_index,
        momentum=0.9997,
        strides=[4, 8, 16, 32],
        norm=False,
        attention=False,
        gradient=True,
        factor=1.0,
    ):
        super(ChangeDetCatBifpnShift, self).__init__(in_channels, num_channels)
        self.in_channels = in_channels
        self.in_index = in_index
        # self.flow_make = nn.Conv2d(in_channels[in_index] * 2 , 2, kernel_size=7, padding=1, bias=False)
        if not norm:
            self.flow_make = nn.Sequential(
                nn.Conv2d(
                    in_channels[in_index] * 2,
                    in_channels[in_index],
                    kernel_size=7,
                    padding=3,
                    bias=True,
                ),
                # nn.Conv2d(in_channels[in_index], in_channels[in_index], kernel_size=7, padding=3, bias=True),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels[in_index], 2, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            self.flow_make = nn.Sequential(
                nn.Conv2d(
                    in_channels[in_index] * 2,
                    in_channels[in_index],
                    kernel_size=7,
                    padding=3,
                    bias=True,
                ),
                nn.BatchNorm2d(num_features=in_channels[in_index]),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels[in_index], 2, kernel_size=3, padding=1, bias=False
                ),
            )
        if attention:
            self.attention_op = nn.Sequential(
                nn.Conv2d(
                    in_channels[in_index] * 2, 1, kernel_size=7, padding=3, bias=True
                ),
                # nn.Conv2d(in_channels[in_index], in_channels[in_index], kernel_size=7, padding=3, bias=True),
                nn.Sigmoid(),
            )
        self.stride = strides[in_index]
        self.gradient = gradient
        self.factor = factor
        self.attention = attention

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid - flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x):
        # warp
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        size_ratio = [
            torch.tensor(x1[i].shape[2:]) / torch.tensor(x1[self.in_index].shape[2:])
            for i in range(len(x1))
        ]
        flow = self.flow_make(torch.cat([x1[self.in_index], x2[self.in_index]], dim=1))
        flow *= self.factor

        if not self.attention:
            flow_mean = flow.mean(dim=(2, 3))
        else:
            att = self.attention_op(
                torch.cat([x1[self.in_index], x2[self.in_index]], dim=1)
            )
            flow_mean = (att * flow).sum(dim=(2, 3)) / att.sum(dim=(2, 3))
        flow_mean = flow_mean[..., None, None].repeat(
            1, 1, x1[self.in_index].shape[2], x1[self.in_index].shape[3]
        )

        x1_out = []
        for i, _x1 in enumerate(x1):
            size = _x1.shape[2:]
            _flow = F.interpolate(flow_mean, size, mode="bilinear", align_corners=True)
            _flow = _flow * size_ratio[i][:, None, None].type_as(_x1).to(_x1.device)
            warp = self.flow_warp(_x1, _flow, size)
            # warp = _x1
            x1_out.append(warp)
        x = [x1_out, x2]

        # bifpn
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = torch.cat((e2x1, e2x2), 1)
        c3 = torch.cat((e3x1, e3x2), 1)
        c4 = torch.cat((e4x1, e4x2), 1)
        c5 = torch.cat((e5x1, e5x2), 1)

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        return fuse

    def forward_loss(self, x, shift):
        # warp
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        size_ratio = [
            torch.tensor(x1[i].shape[2:]) / torch.tensor(x1[self.in_index].shape[2:])
            for i in range(len(x1))
        ]
        flow = self.flow_make(torch.cat([x1[self.in_index], x2[self.in_index]], dim=1))
        flow *= self.factor

        # warp loss
        if not self.attention:
            flow_mean = flow.mean(dim=(2, 3))
            loss_shift = torch.abs(
                flow_mean - (shift / self.stride).squeeze(1).squeeze(1)
            ).mean()
        else:
            att = self.attention_op(
                torch.cat([x1[self.in_index], x2[self.in_index]], dim=1)
            )
            flow_mean = (att * flow).sum(dim=(2, 3)) / att.sum(dim=(2, 3))
            loss_shift = torch.abs(
                flow_mean - (shift / self.stride).squeeze(1).squeeze(1)
            ).mean()

        loss_dict = {"loss_shift": loss_shift}
        flow_mean = flow_mean[..., None, None].repeat(
            1, 1, x1[self.in_index].shape[2], x1[self.in_index].shape[3]
        )

        x1_out = []
        for i, _x1 in enumerate(x1):
            size = _x1.shape[2:]
            if self.gradient:
                _flow = flow_mean
            else:
                _flow = flow_mean.detach()
            _flow = F.interpolate(_flow, size, mode="bilinear", align_corners=True)
            _flow = _flow * size_ratio[i][:, None, None].type_as(_x1).to(_x1.device)
            warp = self.flow_warp(_x1, _flow, size)
            x1_out.append(warp)
        x = [x1_out, x2]

        # bifpn
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = torch.cat((e2x1, e2x2), 1)
        c3 = torch.cat((e3x1, e3x2), 1)
        c4 = torch.cat((e4x1, e4x2), 1)
        c5 = torch.cat((e5x1, e5x2), 1)

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        # out = self.b1out(fuse)
        # return [p7_out, p6_out, p5_out, p4_out, p3_out, p2_out]
        return fuse, loss_dict


@NECKS.register_module()
class ChangeDetCatBifpnShiftNoWarp(ChangeDetCatBifpn):
    def __init__(
        self,
        in_channels,
        num_channels,
        in_index,
        momentum=0.9997,
        strides=[4, 8, 16, 32],
    ):
        super(ChangeDetCatBifpnShiftNoWarp, self).__init__(in_channels, num_channels)
        self.in_channels = in_channels
        self.in_index = in_index
        self.flow_make = nn.Conv2d(
            in_channels[in_index] * 2, 2, kernel_size=7, padding=1, bias=False
        )
        self.flow_make = nn.Sequential(
            nn.Conv2d(
                in_channels[in_index] * 2,
                in_channels[in_index],
                kernel_size=7,
                padding=3,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels[in_index], 2, kernel_size=3, padding=1, bias=False),
        )
        self.stride = strides[in_index]

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid - flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x):
        # warp
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        size_ratio = [
            torch.tensor(x1[i].shape[2:]) / torch.tensor(x1[self.in_index].shape[2:])
            for i in range(len(x1))
        ]
        flow = self.flow_make(torch.cat([x1[self.in_index], x2[self.in_index]], dim=1))
        # flow = self.flow_make(x1[self.in_index] + x2[self.in_index])

        x1_out = []
        for i, _x1 in enumerate(x1):
            size = _x1.shape[2:]
            _flow = F.interpolate(flow, size, mode="bilinear", align_corners=True)
            _flow = _flow * size_ratio[i][:, None, None].type_as(_x1).to(_x1.device)
            # warp = self.flow_warp(_x1, _flow, size)
            warp = _x1
            x1_out.append(warp)
        x = [x1_out, x2]

        # bifpn
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = torch.cat((e2x1, e2x2), 1)
        c3 = torch.cat((e3x1, e3x2), 1)
        c4 = torch.cat((e4x1, e4x2), 1)
        c5 = torch.cat((e5x1, e5x2), 1)

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        # out = self.b1out(fuse)
        # return [p7_out, p6_out, p5_out, p4_out, p3_out, p2_out]
        return fuse

    def forward_loss(self, x, shift):
        # warp
        x1, x2 = x
        assert len(x1) == len(self.in_channels)
        size_ratio = [
            torch.tensor(x1[i].shape[2:]) / torch.tensor(x1[self.in_index].shape[2:])
            for i in range(len(x1))
        ]
        flow = self.flow_make(torch.cat([x1[self.in_index], x2[self.in_index]], dim=1))
        # flow = self.flow_make(x1[self.in_index] + x2[self.in_index])

        x1_out = []
        for i, _x1 in enumerate(x1):
            size = _x1.shape[2:]
            _flow = F.interpolate(flow, size, mode="bilinear", align_corners=True)
            _flow = _flow * size_ratio[i][:, None, None].type_as(_x1).to(_x1.device)
            # warp = self.flow_warp(_x1, _flow, size)
            warp = _x1
            x1_out.append(warp)
        x = [x1_out, x2]

        # warp loss
        flow_gt = (shift / self.stride).squeeze(1).squeeze(1)[..., None, None]
        loss_shift = torch.abs(flow - flow_gt).mean()
        loss_dict = {"loss_shift": loss_shift}

        # bifpn
        x1, x2 = x
        e2x1, e3x1, e4x1, e5x1 = x1
        e2x2, e3x2, e4x2, e5x2 = x2

        c2 = torch.cat((e2x1, e2x2), 1)
        c3 = torch.cat((e3x1, e3x2), 1)
        c4 = torch.cat((e4x1, e4x2), 1)
        c5 = torch.cat((e5x1, e5x2), 1)

        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        p2_out, p3_out, p4_out, p5_out, p6_out, p7_out = self.bifpn(
            [p2, p3, p4, p5, p6, p7]
        )
        p2_out = self.out2(p2_out)
        p3_out = self.out3(p3_out)
        p4_out = self.out4(p4_out)
        p5_out = self.out5(p5_out)
        p6_out = self.out6(p6_out)
        p7_out = self.out7(p7_out)
        fuse = torch.cat((p7_out, p6_out, p5_out, p4_out, p3_out, p2_out), 1)
        # out = self.b1out(fuse)
        # return [p7_out, p6_out, p5_out, p4_out, p3_out, p2_out]
        return fuse, loss_dict
