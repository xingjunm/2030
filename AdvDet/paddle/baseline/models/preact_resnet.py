'''Pre-activation ResNet in PaddlePaddle.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class PreActBlock(nn.Layer):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 nn.BatchNorm2D 默认参数 (例如 momentum=0.9, eps=1e-5) 可能与 PyTorch 的 nn.BatchNorm2d (momentum=0.1, eps=1e-5) 不同。
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 nn.BatchNorm2D 默认参数 (例如 momentum=0.9, eps=1e-5) 可能与 PyTorch 的 nn.BatchNorm2d (momentum=0.1, eps=1e-5) 不同。
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False)
            )

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN, inf)时行为不同。
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN, inf)时行为不同。
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Layer):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 nn.BatchNorm2D 默认参数 (例如 momentum=0.9, eps=1e-5) 可能与 PyTorch 的 nn.BatchNorm2d (momentum=0.1, eps=1e-5) 不同。
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 nn.BatchNorm2D 默认参数 (例如 momentum=0.9, eps=1e-5) 可能与 PyTorch 的 nn.BatchNorm2d (momentum=0.1, eps=1e-5) 不同。
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 nn.BatchNorm2D 默认参数 (例如 momentum=0.9, eps=1e-5) 可能与 PyTorch 的 nn.BatchNorm2d (momentum=0.1, eps=1e-5) 不同。
        self.bn3 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion*planes, kernel_size=1, bias_attr=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False)
            )

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN, inf)时行为不同。
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN, inf)时行为不同。
        out = self.conv2(F.relu(self.bn2(out)))
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN, inf)时行为不同。
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # RISK_INFO: [API 行为不等价] - F.avg_pool2d 在不同框架下的舍入或边界处理行为可能存在细微差异。
        out = F.avg_pool2d(out, 4)
        # CRITICAL_ERROR: [算法流程改变] - 原架构使用 out.view(out.size(0), -1) 进行张量变形，新架构使用了 paddle.flatten 或 out.reshape，API名称、参数获取方式均不一致。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原架构使用 out.view(out.size(0), -1) 进行张量变形，新架构使用了 paddle.flatten 或 out.reshape，API名称、参数获取方式均不一致。")
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((paddle.randn([1,3,32,32])))
    # CRITICAL_ERROR: [命名不一致] - 原架构使用 .size() 方法获取张量尺寸，新架构使用了 .shape 属性。
    # 不一致的实现:
    # print(y.shape)
    # 原架构中的对应代码:
    # print(y.size())
    raise NotImplementedError("原架构使用 .size() 方法获取张量尺寸，新架构使用了 .shape 属性。")

# test()