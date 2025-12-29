'''MobileNet in PaddlePaddle.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Block(nn.Layer):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv2 = nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_planes)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(self.bn1(self.conv1(x)))
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Layer):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同，请确认。
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        # RISK_INFO: [API 行为不等价] - PyTorch F.avg_pool2d 的 count_include_pad 默认为 True，而 Paddle F.avg_pool2d 的 exclusive 默认为 True (等效于 count_include_pad=False)，可能导致计算结果不一致。
        out = F.avg_pool2d(out, 2)
        # CRITICAL_ERROR: [命名不一致] - 原架构使用 out.view(out.size(0), -1) 进行张量变形，新架构使用了 out.reshape([out.shape[0], -1])。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原架构使用 out.view(out.size(0), -1) 进行张量变形，新架构使用了 out.reshape([out.shape[0], -1])。")
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    # RISK_INFO: [随机性处理差异] - paddle.randn 和 torch.randn 的随机数生成器状态和算法不同，即使设置相同的种子，结果也无法复现。
    x = paddle.randn([1,3,32,32])
    y = net(x)
    # CRITICAL_ERROR: [命名不一致] - 原架构使用 y.size() 获取张量尺寸，新架构使用了 y.shape 属性。
    # 不一致的实现:
    # print(y.shape)
    # 原架构中的对应代码:
    # print(y.size())
    raise NotImplementedError("原架构使用 y.size() 获取张量尺寸，新架构使用了 y.shape 属性。")

# test()