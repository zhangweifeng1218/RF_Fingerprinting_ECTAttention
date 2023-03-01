import torch.nn as nn
import math
from model.EfficientChannelSpatialAttention import ecsa_layer

cgmps = []
cgaps = []
tgmps = []
tgaps = []
channelAttMaps = []
temporalAttMaps = []

# 这是残差网络中的basicblock，实现的功能如下方解释：
class ECSABasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
    # inplanes代表输入通道数，planes代表输出通道数。
        super(ECSABasicBlock, self).__init__()

        # Conv1
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv2
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.aca = ecsa_layer(planes, k_size)
        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out, (intermedia) = self.aca(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ECSABottleneck(nn.Module):
    expansion = 4      # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECSABottleneck, self).__init__()

        # conv1   1x1
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # conv2   3x3
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=True, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        # conv3   1x1
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.aca = ecsa_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out, (intermedia) = self.aca(out)
        channelAttMaps.append(intermedia[0])
        temporalAttMaps.append(intermedia[1])
        cgaps.append(intermedia[2])
        cgmps.append(intermedia[3])
        tgaps.append(intermedia[4])
        tgmps.append(intermedia[5])


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECSA_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        # layers=参数列表 block选择不同的类
        self.inplanes = 64
        super(ECSA_ResNet, self).__init__()
        # 1.conv1
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((512, block.expansion))#nn.AvgPool1d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


# resnet18
def ecsa_resnet18(cfg):
    num_classes = cfg['n_classes']
    model = ECSA_ResNet(ECSABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=[-3, -3, -3, -3])
    return model

# resnet34
def ecsa_resnet34(cfg):
    num_classes = cfg['n_classes']
    model = ECSA_ResNet(ECSABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=[-3, -3, -3, -3])
    return model

# resnet50
def ecsa_resnet50(cfg):
    num_classes = cfg['n_classes']
    model = ECSA_ResNet(ECSABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=[-3, -3, -3, -3])
    return model

# resnet101
def ecsa_resnet101(cfg):
    num_classes = cfg['n_classes']
    model = ECSA_ResNet(ECSABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=[-3, -3, -3, -3])
    return model

# resnet152
def ecsa_resnet152(cfg):
    num_classes = cfg['n_classes']
    model = ECSA_ResNet(ECSABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=[-3, -3, -3, -3])
    return model

if __name__=='__main__':
    from config import cfg
    import torch
    model = ecsa_resnet50(cfg)


    input = torch.randn(1, 2, 512)
    out = model(input)
    print(out.shape)
    from utils import count_parameters
    print(count_parameters(model))
