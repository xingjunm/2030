'''DenseNet in PaddlePaddle.'''
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Bottleneck(nn.Layer):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 Conv2D 默认使用 Xavier 初始化，而 PyTorch 的 Conv2d 默认使用 Kaiming Uniform 初始化。这会导致模型初始权重不同。
        self.conv1 = nn.Conv2D(in_planes, 4*growth_rate, kernel_size=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(4*growth_rate)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 Conv2D 默认使用 Xavier 初始化，而 PyTorch 的 Conv2d 默认使用 Kaiming Uniform 初始化。这会导致模型初始权重不同。
        self.conv2 = nn.Conv2D(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理边缘情况(如NaN, inf)时行为不同。
        out = self.conv1(F.relu(self.bn1(x)))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理边缘情况(如NaN, inf)时行为不同。
        out = self.conv2(F.relu(self.bn2(out)))
        # RISK_INFO: [张量操作差异] - paddle.concat 与 torch.cat 功能相似，但底层实现和性能特性可能不同。
        out = paddle.concat([out,x], 1)
        return out


class Transition(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2D(in_planes)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 Conv2D 默认使用 Xavier 初始化，而 PyTorch 的 Conv2d 默认使用 Kaiming Uniform 初始化。这会导致模型初始权重不同。
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理边缘情况(如NaN, inf)时行为不同。
        out = self.conv(F.relu(self.bn(x)))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.avg_pool2d 的默认行为(exclusive=True)与 torch.nn.functional.avg_pool2d(默认 count_include_pad=True)在处理padding区域时不同。
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Layer):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 Conv2D 默认使用 Xavier 初始化，而 PyTorch 的 Conv2d 默认使用 Kaiming Uniform 初始化。这会导致模型初始权重不同。
        self.conv1 = nn.Conv2D(3, num_planes, kernel_size=3, padding=1, bias_attr=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2D(num_planes)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 Linear 默认使用 Xavier 初始化，而 PyTorch 的 Linear 默认使用 Kaiming Uniform 初始化。这会导致模型初始权重不同。
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.avg_pool2d 的默认行为(exclusive=True)与 torch.nn.functional.avg_pool2d(默认 count_include_pad=True)在处理padding区域时不同。
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        # CRITICAL_ERROR: [API 行为不等价] - 原架构使用 out.view()，新架构使用 out.reshape()。PyTorch的 view() 要求张量是内存连续的，而 reshape() 更灵活，这是API行为上的关键不一致。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原架构使用 out.view()，新架构使用 out.reshape()。PyTorch的 view() 要求张量是内存连续的，而 reshape() 更灵活，这是API行为上的关键不一致。")
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    # RISK_INFO: [随机性处理差异] - paddle.randn 与 torch.randn 的随机数生成器状态和算法不同，即使设置相同的seed，结果也无法复现。
    x = paddle.randn([1,3,32,32])
    y = net(x)
    print(y)

# test()