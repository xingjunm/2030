'''Simplified version of DLA in PaddlePaddle.

Note this implementation is not identical to the original paper version.
But it seems works fine.

See dla.py for the original paper version.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 BatchNorm2D 默认 momentum 为 0.9，而PyTorch默认为 0.1。这是一个常见的迁移陷阱，可能导致训练动态不一致。
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 BatchNorm2D 默认 momentum 为 0.9，而PyTorch默认为 0.1。这是一个常见的迁移陷阱，可能导致训练动态不一致。
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias_attr=False),
                # RISK_INFO: [API 行为不等价] - PaddlePaddle的 BatchNorm2D 默认 momentum 为 0.9，而PyTorch默认为 0.1。这是一个常见的迁移陷阱，可能导致训练动态不一致。
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(out)
        return out


class Root(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias_attr=False)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 BatchNorm2D 默认 momentum 为 0.9，而PyTorch默认为 0.1。这是一个常见的迁移陷阱，可能导致训练动态不一致。
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, xs):
        x = paddle.concat(xs, 1)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Layer):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class SimpleDLA(nn.Layer):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        # CRITICAL_ERROR: [预设内容不一致] - 原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。
        # 不一致的实现:
        # self.base = nn.Sequential(
        #     nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias_attr=False),
        #     nn.BatchNorm2D(16),
        #     nn.ReLU()
        # )
        # 原架构中的对应代码:
        # self.base = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True)
        # )
        raise NotImplementedError("原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。")

        # CRITICAL_ERROR: [预设内容不一致] - 原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。
        # 不一致的实现:
        # self.layer1 = nn.Sequential(
        #     nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1, bias_attr=False),
        #     nn.BatchNorm2D(16),
        #     nn.ReLU()
        # )
        # 原架构中的对应代码:
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True)
        # )
        raise NotImplementedError("原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。")

        # CRITICAL_ERROR: [预设内容不一致] - 原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。
        # 不一致的实现:
        # self.layer2 = nn.Sequential(
        #     nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1, bias_attr=False),
        #     nn.BatchNorm2D(32),
        #     nn.ReLU()
        # )
        # 原架构中的对应代码:
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True)
        # )
        raise NotImplementedError("原作使用 nn.ReLU(True)，新实现为 nn.ReLU()，缺少 inplace=True 参数。")

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.avg_pool2d 默认 exclusive=True (不计算 padding)，而PyTorch默认为 count_include_pad=True。虽然在此处可能无影响，但在其他情况下可能导致结果差异。
        out = F.avg_pool2d(out, 4)
        # CRITICAL_ERROR: [命名不一致] - 原作使用 .view(out.size(0), -1)，新实现使用了 .reshape 和 .shape。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原作使用 .view(out.size(0), -1)，新实现使用了 .reshape 和 .shape。")
        out = self.linear(out)
        return out


def test():
    net = SimpleDLA()
    print(net)
    x = paddle.randn([1, 3, 32, 32])
    y = net(x)
    # CRITICAL_ERROR: [命名不一致] - 获取尺寸的方法在原作中为 '.size()'，而不是属性 '.shape'。
    # 不一致的实现:
    # print(y.shape)
    # 原架构中的对应代码:
    # print(y.size())
    raise NotImplementedError("获取尺寸的方法在原作中为 '.size()'，而不是属性 '.shape'。")


if __name__ == '__main__':
    test()