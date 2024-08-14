from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
import cv2

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .base_model import BaseModel

try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None
from dcn.modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        if 'gen4' in opt.dataset:
            inp_channels = 10
            self.event_data = True
        else:
            inp_channels = 3
            self.event_data = False

        self.base_layer = nn.Sequential(
            nn.Conv2d(inp_channels, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        self.pre_img_layer = None
        if opt.pre_img and not opt.is_recurrent:
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.pre_hm_layer = None
        if opt.pre_hm and not opt.is_recurrent:
            self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                    padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None and self.pre_img_layer is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None and self.pre_hm_layer is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    else:
        print('Warning: No ImageNet pretrain!!')
    return model

def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla102', hash='d94d9790')
    return model

def dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60', hash='24839fc4')
    return model


def dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(
            data='imagenet', name='dla169', hash='0914e092')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)),
            nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False, 
                                dilation=d, padding=(0, d * (k // 2))),
            nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False, 
                                dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, 
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
    'gcn': (Conv, GlobalConv),
    'conv': (Conv, Conv),
}

class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio=4
        self.opt = opt
        self.dcn_3d_aggregate = opt.dcn_3d_aggregation
        self.conv_3d_aggregate = opt.conv_3d_aggregation
        assert not (self.dcn_3d_aggregate and self.conv_3d_aggregate)
        self.save_videos_dcn3d = opt.save_videos_dcn3d
        if self.save_videos_dcn3d:
            self.video_dcn3d_dir = opt.video_dcn3d_dir
        self.temporal_aggregate = opt.temporal_aggregation

        self.node_type = DLA_NODE[opt.dla_node]
        print('Using node type:', self.node_type)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)

        if self.dcn_3d_aggregate:
            kT, kH, kW = 2, 3, 3
            sT, sH, sW = 1, 1, 1
            pT, pH, pW = 0, 1, 1
            self.dcn_3d = DeformConvPack_d(10, 10, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension='THW')
            self.dcn_history = None
            self.aggr_history = None

        if self.conv_3d_aggregate:
            kT, kH, kW = 2, 3, 3
            sT, sH, sW = 1, 1, 1
            pT, pH, pW = 0, 1, 1
            self.dcn_3d = nn.Conv3d(10, 10, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW])
            self.dcn_history = None
            self.aggr_history = None

        if self.temporal_aggregate:
            assert self.dcn_3d_aggregate or self.conv_3d_aggregate
            self.conv2d_1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, bias=True)
            self.conv2d_2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.timemap = None

    def freeze_backbone(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False    

        for parameter in self.dla_up.parameters():
            parameter.requires_grad = False

        for parameter in self.ida_up.parameters():
            parameter.requires_grad = False  

    def do_tensor_pass(self, x, pre_img, pre_hm):
        x = self.base(x, pre_img, pre_hm) # TODO
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def dcn_3d_aggregation(self, inp, inp_history):
        B, T, C, H, W = inp.shape
        inp = inp.view(B, C, T, H, W)
        inp_history = inp_history.view(B, C, T, H, W)
        inp_cat = torch.cat((inp, inp_history), 2)
        out = self.dcn_3d(inp_cat)
        out = out.view(B, T, C, H, W)
        return out

    def temporal_aggregation(self, inp, inp_history):
        mask = (inp!=0)
        mask = (torch.sum(mask, dim=2, keepdim=True) > 0)
        aggr_history = torch.logical_not(mask) * inp_history + mask * inp
        return aggr_history

    def feature_map_to_image(self, input, feature_map, frame_counts, delta_t=50000, choice='all'):
        C, H, W = feature_map.shape
        feature_map = feature_map.cpu().data.numpy()
        choices = ['all', 'range','max', 'bin', 'gray', 'pos', 'neg']
        assert choice in choices

        input = input.cpu().data.numpy()
        input_mask = (input!=0)                         # shape = [C, H, W]
        input_mask = (np.sum(input_mask, axis=0) > 0)    # shape = [H, W]
        input_mask = input_mask.astype('uint8') * 255

        if choice == 'range':
            image = np.zeros((H*10, W*10, 3) ,dtype=np.uint8)

            for c in range(C):
                range_min = np.min(feature_map[c, ...])
                range_max = np.max(feature_map[c, ...])
                interval = (range_max - range_min) / 10
                for i in range(10):
                    mask = ((feature_map[c, ...]>=range_min+interval*i) & (feature_map[c, ...]<=range_min+interval*(i+1))).astype('uint8') * 255
                    row = c
                    col = i
                    image[row*H:(row+1)*H, col*W:(col+1)*W, :] = np.stack([mask]*3, axis=2).copy()
                    cv2.putText(image[row*H:(row+1)*H, col*W:(col+1)*W, :], '{} ms / {} ms'.format(frame_counts*delta_t//1000, (frame_counts+1)*delta_t//1000), 
                            (W-200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    if c == C -1:
                        image = cv2.line(image, (W*i,0), (W*i,H*10), (0,0,255), 2)
                image = cv2.line(image, (0,H*c), (W*10,H*c), (0,0,255), 2)
            image = cv2.line(image, (0,H*C), (W*10,H*C), (0,0,255), 1)
            image = cv2.line(image, (W*10,0), (W*10,H*10), (0,0,255), 2)
        else:
            image = np.zeros((H*4, W*3, 3) ,dtype=np.uint8)

            if choice == 'max':
                m = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                max_map = m(torch.from_numpy(feature_map))
                max_map = max_map.numpy()

            for c in range(C):
                if choice == 'all':
                    feature_map[c, ...] = np.abs(feature_map[c, ...])
                    fmin = np.min(feature_map[c, ...])
                    fmax = np.max(feature_map[c, ...])
                    mask = (feature_map[c, ...] - fmin) / (fmax - fmin) * 255
                    mask = mask.astype('uint8')
                    clahe = cv2.createCLAHE()
                    mask = clahe.apply(mask)
                elif choice == 'max':
                    mask = (feature_map[c, ...] == max_map[c, ...]).astype('uint8') * 255
                elif choice == 'bin':
                    # mask = (feature_map[c, ...] != 0).astype('uint8') * 255 # shape = (H, W)
                    mask = (np.abs(feature_map[c, ...]) > 0.01).astype('uint8') * 255 # shape = (H, W)
                    # model_54 0
                    # model_82 0.03
                    # model_71 0.05
                elif choice == 'gray':
                    feature_map[c, ...] = feature_map[c, ...] * (np.abs(feature_map[c, ...]) >= 0.05)
                    feature_map[c, ...] = np.abs(feature_map[c, ...])
                    fmin = np.min(feature_map[c, ...])
                    fmax = np.max(feature_map[c, ...])
                    mask = (feature_map[c, ...] - fmin) / (fmax - fmin) * 255
                elif choice == 'pos':
                    feature_map[c, ...] = feature_map[c, ...] * (feature_map[c, ...] >= 0)
                    fmin = np.min(feature_map[c, ...])
                    fmax = np.max(feature_map[c, ...])
                    mask = (feature_map[c, ...] - fmin) / (fmax - fmin) * 255
                    mask = mask.astype('uint8')
                    clahe = cv2.createCLAHE()
                    mask = clahe.apply(mask)
                elif choice == 'neg':
                    feature_map[c, ...] = feature_map[c, ...] * (feature_map[c, ...] < 0)
                    fmin = np.min(feature_map[c, ...])
                    fmax = np.max(feature_map[c, ...])
                    mask = 255 - (feature_map[c, ...] - fmin) / (fmax - fmin) * 255
                    mask = mask.astype('uint8')
                    clahe = cv2.createCLAHE()
                    mask = clahe.apply(mask)
                mask = mask.astype('uint8')

                row = c // 3
                col = c % 3
                image[row*H:(row+1)*H, col*W:(col+1)*W, :] = np.stack([mask,mask,mask], axis=2).copy()
                cv2.putText(image[row*H:(row+1)*H, col*W:(col+1)*W, :], '{} ms / {} ms'.format(frame_counts*delta_t//1000, (frame_counts+1)*delta_t//1000), 
                        (W-200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if c == C-1:
                    ret_image = np.stack([mask[12:-12, :]], axis=2).copy()
            col = 2
            image[row*H:(row+1)*H, col*W:(col+1)*W, :] = np.stack([input_mask]*3, axis=2).copy()

            image = cv2.line(image, (0,H*1), (W*3,H*1), (0,0,255), 1)
            image = cv2.line(image, (0,H*2), (W*3,H*2), (0,0,255), 1)
            image = cv2.line(image, (0,H*3), (W*3,H*3), (0,0,255), 1)
            image = cv2.line(image, (W*1,0), (W*1,H*4), (0,0,255), 1)
            image = cv2.line(image, (W*2,0), (W*2,H*4), (0,0,255), 1)

        return image, ret_image

    def imgpre2feats(self, x, pre_img=None, pre_hm=None, reset_dcn_3d=False, video_names=[]):
        if self.opt.is_recurrent:            
            if type(x) == list:
                if 'depth' in x[0] and x[0]['depth'] is not None:
                    depth_inp = x[0]['depth'].unsqueeze(1)
                    for i in range(1, len(x)):
                        depth_inp = torch.cat((depth_inp, x[i]['depth'].unsqueeze(1)), 1)
                    inp_len = 1
                    if not self.opt.stream_test:
                        # inp_len = self.opt.input_len
                        inp_len = len(x)
                    pre_img = depth_inp.view(len(x[0]['depth']) * inp_len, x[0]['depth'].size(1), x[0]['depth'].size(2), x[0]['depth'].size(3))

                if reset_dcn_3d:
                    B, C, H, W = x[0]['image'].shape
                    T = 1
                    C = 1
                    self.frame_counts = torch.tensor([0] * B)

                if self.dcn_3d_aggregate or self.conv_3d_aggregate:
                    if reset_dcn_3d:
                        self.dcn_history = None
                        self.aggr_history = None

                    if self.temporal_aggregate:
                        inp_history = None
                        inp_history = x[0]['image'].unsqueeze(1)
                        for i in range(1, len(x)):
                            inp_history = torch.cat((inp_history, x[i]['image'].unsqueeze(1)), 1)

                    if self.dcn_history == None:
                        inp = x[0]['image'].unsqueeze(1)
                        self.dcn_history = inp.clone()

                        if self.temporal_aggregate:
                            aggr_history = inp.clone()
                            self.aggr_history = inp.clone()

                        for i in range(1, len(x)):
                            self.dcn_history = self.dcn_3d_aggregation(x[i]['image'].unsqueeze(1), self.dcn_history.detach())
                            inp = torch.cat((inp, self.dcn_history), 1)
                            if self.temporal_aggregate:
                                self.aggr_history = self.temporal_aggregation(x[i]['image'].unsqueeze(1), self.aggr_history.detach())
                                aggr_history = torch.cat((aggr_history, self.aggr_history), 1)
                    else:
                        inp = None
                        aggr_history = None
                        for i in range(0, len(x)):
                            self.dcn_history = self.dcn_3d_aggregation(x[i]['image'].unsqueeze(1), self.dcn_history.detach())
                            if inp == None:
                                inp = self.dcn_history.clone()
                            else:
                                inp = torch.cat((inp, self.dcn_history), 1)

                            if self.temporal_aggregate:
                                self.aggr_history = self.temporal_aggregation(x[i]['image'].unsqueeze(1), self.aggr_history.detach())
                                if aggr_history == None:
                                    aggr_history = self.aggr_history.clone()
                                else:
                                    aggr_history = torch.cat((aggr_history, self.aggr_history), 1)
                else:
                    inp = x[0]['image'].unsqueeze(1)
                    for i in range(1, len(x)):
                        inp = torch.cat((inp, x[i]['image'].unsqueeze(1)), 1)

                inp_len = 1
                if not self.opt.stream_test:
                    # inp_len = self.opt.input_len
                    inp_len = len(x)

                if self.temporal_aggregate:
                    B, T, C, H, W = inp.shape
                    inp = inp.view(B*T, C, H, W)
                    inp = self.conv2d_1(inp)
                    inp = self.conv2d_2(inp)
                    inp = inp.view(B, T, 1, H, W)
                    alpha = self.sigmoid(inp)
                    inp = alpha * inp_history + (1 - alpha) * aggr_history

                if self.save_videos_dcn3d:
                    choice = 'all'
                    if reset_dcn_3d:
                        B, C, H, W = x[0]['image'].shape
                        self.video_paths = [os.path.join(self.video_dcn3d_dir, video_name + '.avi') for video_name in video_names]
                        self.image_paths = [os.path.join(self.video_dcn3d_dir, video_name + '_{}.jpg') for video_name in video_names]
                        if choice == 'range':
                            self.video_files = [cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (W*10, H*10)) for video_path in self.video_paths]
                        else:
                            self.video_files = [cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (W*3, H*4)) for video_path in self.video_paths]

                    for b in range(inp.size(0)):
                        for t in range(inp.size(1)):
                            image, ret_image = self.feature_map_to_image(x[t]['image'][b], inp[b, t, ...], self.frame_counts[b], choice = choice)
                            self.video_files[b].write(image)
                            cv2.imwrite(self.image_paths[b].format(self.frame_counts[b]), ret_image)
                            self.frame_counts[b] = self.frame_counts[b] + 1
                            if self.frame_counts[b] == 1200:
                                self.video_files[b].release()
                else:
                    self.frame_counts = self.frame_counts + 1

                x = inp.view(len(x[0]['image']) * inp_len, x[0]['image'].size(1), x[0]['image'].size(2), x[0]['image'].size(3))
            else:
                x = torch.stack((pre_img, x), 1).view(len(pre_img) * 2, pre_img.size(1), pre_img.size(2), pre_img.size(3))

        if self.opt.batch_split_factor == -1:
            y = self.do_tensor_pass(x, pre_img, pre_hm)
        else:
            step = int(round(len(x) / self.opt.batch_split_factor))
            feat_maps = []
            count = 1
            for i in range(0, len(x), step):
                y = self.do_tensor_pass(x[i: i + step], pre_img, pre_hm)
                feat_maps.append(y.detach())
                count += 1
            y = torch.cat(feat_maps, 0)

        return [y]
