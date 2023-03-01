import torch.nn as nn
import torch.nn.functional as F
import math
from model.EfficientChannelSpatialAttention import ecsa_layer


class BN_Conv1d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv1d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))


class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv1d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv1d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv1d(self.group_chnls, self.group_chnls*2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(self.group_chnls*2)
        self.short_cut = nn.Sequential(
            nn.Conv1d(in_chnls, self.group_chnls*2, 1, stride, 0, bias=False),
            nn.BatchNorm1d(self.group_chnls*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes) -> object:
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv1d(2, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)   # 224x224 input size

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool1d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.adaptive_max_pool2d(out, (512, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resNeXt50(cfg):
    num_classes = cfg['n_classes']
    return ResNeXt([3, 4, 6, 3], 20, 4, num_classes)


def resNeXt101(cfg):
    num_classes = cfg['n_classes']
    return ResNeXt([3, 4, 23, 3], 20, 4, num_classes)



if __name__=='__main__':
    from config import cfg
    import torch
    model = resNeXt101(cfg)


    input = torch.randn(1, 2, 512)
    out = model(input)
    print(out.shape)
    from utils import count_parameters
    print(count_parameters(model))