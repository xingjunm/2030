'''DLA in PaddlePaddle.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def __view_compatibility(tensor, *shape):
    """
    Framework compatibility function for view() operation.
    PyTorch's view() requires contiguous tensors while PaddlePaddle's reshape() is more flexible.
    """
    return tensor.reshape(shape)


def __size_compatibility(tensor, dim=None):
    """
    Framework compatibility function for size() method.
    PyTorch uses .size() while PaddlePaddle uses .shape
    """
    if dim is None:
        return tensor.shape
    return tensor.shape[dim]


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, xs):
        x = paddle.concat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Layer):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA(nn.Layer):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(DLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(16),
            # CRITICAL_ERROR: [预设内容不一致] - ReLU层缺少 'inplace=True' 参数，与原作不一致。
            # 不一致的实现:
            # nn.ReLU()
            # 原架构中的对应代码:
            # nn.ReLU(True)
            raise NotImplementedError("ReLU层缺少 'inplace=True' 参数，与原作不一致。")
        )

        self.layer1 = nn.Sequential(
            nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(16),
            # CRITICAL_ERROR: [预设内容不一致] - ReLU层缺少 'inplace=True' 参数，与原作不一致。
            # 不一致的实现:
            # nn.ReLU()
            # 原架构中的对应代码:
            # nn.ReLU(True)
            raise NotImplementedError("ReLU层缺少 'inplace=True' 参数，与原作不一致。")
        )

        self.layer2 = nn.Sequential(
            nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(32),
            # CRITICAL_ERROR: [预设内容不一致] - ReLU层缺少 'inplace=True' 参数，与原作不一致。
            # 不一致的实现:
            # nn.ReLU()
            # 原架构中的对应代码:
            # nn.ReLU(True)
            raise NotImplementedError("ReLU层缺少 'inplace=True' 参数，与原作不一致。")
        )

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
        out = F.avg_pool2d(out, 4)
        out = __view_compatibility(out, __size_compatibility(out, 0), -1)
        out = self.linear(out)
        return out


def test():
    net = DLA()
    print(net)
    # RISK_INFO: [随机性处理差异] - paddle.randn 和 torch.randn 的随机数生成器实现和默认种子可能不同，可能导致模型初始化和复现性存在差异。
    x = paddle.randn([1, 3, 32, 32])
    y = net(x)
    print(__size_compatibility(y))


if __name__ == '__main__':
    test()