import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron_kernel


class LIFSpike(nn.Module):
    def __init__(self, tau: float = 2., v_threshold: float = 1., T: int = 4):
        super(LIFSpike, self).__init__()
        self.tau = tau
        self.thresh = v_threshold
        self.act = neuron_kernel.MultiStepLIFNodePTT.apply
        self.T = T

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(self.T, B // self.T, C, H, W)
        shape = x.shape
        v = torch.zeros_like(x[0].data)
        spike_seq, _ = self.act(
            x.flatten(1), v.flatten(0), False, self.tau, self.thresh, 0.,
            True, surrogate.Sigmoid().cuda_code)
        spike_seq = spike_seq.reshape(shape)
        return spike_seq.reshape(-1, C, H, W)


class FCLIFSpike(nn.Module):
    def __init__(self, tau: float = 2., v_threshold: float = 1., T: int = 4):
        super(FCLIFSpike, self).__init__()
        self.tau = tau
        self.thresh = v_threshold
        self.act = neuron_kernel.MultiStepLIFNodePTT.apply
        self.T = T

    def forward(self, x):
        # T, B, C = x.shape
        # x = x.reshape(self.T, B // self.T, C, H, W)
        shape = x.shape
        v = torch.zeros_like(x[0].data)
        spike_seq, _ = self.act(
            x.flatten(1), v.flatten(0), False, self.tau, self.thresh, 0.,
            True, surrogate.Sigmoid().cuda_code)
        spike_seq = spike_seq.reshape(shape)
        return spike_seq.reshape(shape)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = LIFSpike()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = LIFSpike()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out)

        return out + identity


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, g='ADD', down='max', T=4, ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = LIFSpike()

        self.layer1 = self._make_layer(block, 128, layers[0], )
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 256)
        self.fc_spike = FCLIFSpike()
        self.fc_ = nn.Linear(256, num_classes)

        self.T = T

        self.project1 = projector(128 * block.expansion, T)
        self.project2 = projector(256 * block.expansion, T)
        self.project3 = projector(512 * block.expansion, T)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.T = self.T

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                #                 LIFSpike(),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.ModuleList(layers)

    #         return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        B, C, H, W = x.shape
        # #         if self.training:
        # #             B = B // self.T
        # #         else:
        x = x.unsqueeze(0)
        x = x.repeat(self.T, 1, 1, 1, 1).reshape(-1, C, H, W)
        # B, T, C, H, W = x.shape
        # x = x.permute(1, 0, 2, 3, 4)
        # x = x.reshape(-1, C, H, W)



        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        fea = []
        origin_fea = []

        # layer1
        for blk in self.layer1:
            x = blk(x)

        #         x = self.dp1(x)
        if self.training:
            fea.append(self.project1(x))

        if not self.training:
            origin_fea.append(x)
        #         print(x)
        # layer2
        for blk in self.layer2:
            x = blk(x)

        #         x = self.dp2(x)
        if self.training:
            fea.append(self.project2(x))
        if not self.training:
            origin_fea.append(x)
        #         print(x)
        # layer3
        for blk in self.layer3:
            x = blk(x)

        #         x = self.dp3(x)
        if self.training:
            fea.append(self.project3(x))
        if not self.training:
            origin_fea.append(x)
        #         print(x)
        # final
        x = self.avgpool(x)
        # if not self.training:
        #     origin_fea.append(x)
        x = torch.flatten(x, 1).reshape(self.T, B, -1)
        out = self.fc(x)
        out = self.fc_(self.fc_spike(out))
        # if self.training:
        #     return out, fea
        return out.mean(0), out #origin_fea

    def forward(self, x):
        return self._forward_impl(x)


import torch.nn.functional as F


class projector(nn.Module):
    def __init__(self, channel, T=4):
        super(projector, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            #             nn.GroupNorm(1,channel,affine=False)
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),

        )
        #         self.bn = nn.BatchNorm1d(channel)
        self.T = T

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = self.avg(x)
        #         x = self.bn(x.flatten(1))
        #         x = x.reshape(self.T,-1,x.shape[1])
        x = torch.flatten(x, 1).reshape(self.T, B // self.T, -1)
        #         return x
        #         print(x.shape)
        return F.normalize(x, dim=2)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet19_(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)
