
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate ,neuron_kernel, neuron

class TEBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(4, 1, 1, 1, 1))
        self.T = 4

    def forward(self, input):
        # x B,C,H,W - T,B,C,H,W
        B,C,H,W = input.shape
        y = input.reshape(self.T, B // self.T, C, H, W).permute(1,2,0,3,4) # B,C,H,W - T,B,C,H,W - B,C,T,H,W
        # y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        # y = y.contiguous().transpose(1, 2)
        y = y.permute(2,0,1,3,4)  # B,C,T,H,W - T,B,C,H,W
        y = y * self.p
        y = y.reshape(B,C,H,W)  # B,C,H,W
        return y



class LIFSpike(nn.Module):
    def __init__(self ,tau: float = 2., v_threshold: float = 1.0, T: int = 4):
        super(LIFSpike, self).__init__()
        self.tau = tau
        self.thresh = v_threshold
        self.act = neuron_kernel.MultiStepLIFNodePTT.apply
        # self.act = neuron_kernel.MultiStepIFNodePTT.apply
        self.T = T
    def forward(self ,x):
        B, C, H, W = x.shape
        x = x.reshape(self.T, B // self.T, C, H, W)
        shape = x.shape
        v = torch.zeros_like(x[0].data)
        # IF
        # spike_seq, _ = neuron_kernel.MultiStepIFNodePTT.apply(
        #     x.flatten(1), v.flatten(0), self.thresh, None, True,
        #     surrogate.Sigmoid().cuda_code)

        # LIF
        spike_seq, _ = self.act(
            x.flatten(1), v.flatten(0), False, self.tau, self.thresh, 0.,
            True, surrogate.Sigmoid().cuda_code)
        spike_seq = spike_seq.reshape(shape)
        return spike_seq.reshape(-1, C, H, W)

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

        out = self.relu(out)

        return out + identity


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
                 norm_layer=None ,g ='ADD' ,down='max', T=4, ):
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
        # self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = LIFSpike()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #         self.fc_auxiliary = nn.Linear(512 * block.expansion, num_classes * T)

        #         self.project1 = projector(64 * block.expansion,T)
        #         self.project2 = projector(128 * block.expansion,T)
        #         self.project3 = projector(256 * block.expansion,T)
        #         self.project4 = projector(512 * block.expansion,T)

        #         self.dp1 = nn.Dropout(p=0.5)
        #         self.dp2 = nn.Dropout(p=0.5)
        #         self.dp3 = nn.Dropout(p=0.5)
        #         self.dp4 = nn.Dropout(p=0.5)

        self.T = T

        self.project1 = projector(64 * block.expansion, T)
        self.project2 = projector(128 * block.expansion, T)
        self.project3 = projector(256 * block.expansion, T)
        self.project4 = projector(512 * block.expansion, T)

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
                LIFSpike(),
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

        #         if self.training:
        #             B = B // self.T
        #         else:
        # x = x.unsqueeze(0)
        B, C, H, W = x.shape
        x = x.unsqueeze(0)
        x = x.repeat(self.T, 1, 1, 1, 1).reshape(-1, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        fea = []
        origin_fea = []

        # layer1
        for blk in self.layer1:
            x = blk(x)
            if not self.training:
                origin_fea.append(x)
        #         x = self.dp1(x)
        if self.training:
            fea.append(self.project1(x))

        # layer2
        for blk in self.layer2:
            x = blk(x)
            if not self.training:
                origin_fea.append(x)
        #         x = self.dp2(x)
        if self.training:
            fea.append(self.project2(x))

        # layer3
        for blk in self.layer3:
            x = blk(x)
            if not self.training:
                origin_fea.append(x)
        #         x = self.dp3(x)
        if self.training:
            fea.append(self.project3(x))



        # layer4

        for blk in self.layer4:
            x = blk(x)
            if not self.training:
                origin_fea.append(x)
        #         x = self.dp4(x)
        if self.training:
            fea.append(self.project4(x))


        # final
        x = self.avgpool(x)
        if not self.training:
            origin_fea.append(x)
        x = torch.flatten(x, 1).reshape(self.T, B, -1)
        # if not self.training:
        #     origin_fea.append(x.mean(0))
        out = self.fc(x)
        #         out_auxiliary = self.fc_auxiliary(x)

        # if self.training:
        #     return out, fea

        #         out_auxiliary = out_auxiliary.reshape(self.T,B,self.T,-1)
        #         out_auxiliary = out_auxiliary.mean(2)
        return out.mean(0), out

    #         x = self.layer1(x)
    #         if self.training:
    # #             pass
    #             fea.append(self.project1(x))
    #         else:
    #             origin_fea.append(x)

    #         x = self.layer2(x)
    #         if self.training:
    # #             pass
    #             fea.append(self.project2(x))
    #         else:
    #             origin_fea.append(x)

    #         x = self.layer3(x)
    #         if self.training:
    # #             pass
    #             fea.append(self.project3(x))
    #         else:
    #             origin_fea.append(x)

    #         x = self.layer4(x)
    #         if self.training:
    #             fea.append(self.project4(x))
    #         else:
    #             origin_fea.append(x)

    #         x = self.avgpool(x)
    #         if not self.training:
    #             origin_fea.append(x)
    #         x = torch.flatten(x, 1).reshape(self.T, B,-1)
    #         out = self.fc(x)
    #         if self.training:
    #             return out , fea
    #         return out, origin_fea

    def forward(self, x):
        return self._forward_impl(x)


import torch.nn.functional as F


# class projector(nn.Module):
#     def __init__(self,dim,T=4):
#         super(projector, self).__init__()
#         self.project = nn.Sequential(
#             nn.Conv2d(dim, dim, 3,1,1,bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#         )
#         self.predect = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU()
#         )
#         self.T = T
#     def forward(self, x):
#         # B * T , C , H , W = x.shape
#         x = self.project(x)
#         target = x.detach().flatten(2).mean(2).reshape(self.T,x.shape[0] // self.T,-1).permute(1,0,2)
#         predection = self.predect(x).flatten(2).mean(2).reshape(self.T,x.shape[0] // self.T,-1).permute(1,0,2)
#         return F.normalize(predection,dim=-1),F.normalize(target,dim=-1)


class projector(nn.Module):
    def __init__(self, channel, T=4):
        super(projector, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),

        )

        self.T = T

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = self.avg(x)

        x = torch.flatten(x, 1).reshape(self.T, B // self.T, -1)


        return F.normalize(x, dim=2)


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


# class projector(nn.Module):
#     def __init__(self ,channel ,T=4):
#         super(projector, self).__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
# #             nn.BatchNorm2d(channel),
# #             nn.ReLU(),
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
# #             nn.BatchNorm2d(channel),
# #             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
# #             nn.BatchNorm2d(channel),
# #             nn.ReLU(),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel ,channel ,3 ,1 ,1 ,bias=False),
# #             nn.BatchNorm2d(channel),
# #             nn.ReLU(),
#         )
#         self.T = T
#     def forward(self ,x):
#         # B*T,C,H,W
#         B = x.shape[0] // self.T
#         x = torch.cat([self.conv0(x[:B]),self.conv1(x[B:B*2]),self.conv2(x[B*2:B*3]),self.conv3(x[B*3:])],dim=0)
# #         x = self.conv(x)
#         x = self.avg(x)
#         x = torch.flatten(x ,1).reshape(self.T ,-1 ,x.shape[1])
#         return F.normalize(x ,dim=2)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet24(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resne248', BasicBlock, [2, 3, 3, 3], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet40(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet40', BasicBlock, [4, 5, 7, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet56(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet56', Bottleneck, [3, 5, 7, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


# model = resnet18()
# model.fc.parameters()