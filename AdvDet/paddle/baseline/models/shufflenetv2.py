'''ShuffleNetV2 in PaddlePaddle.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ShuffleBlock(nn.Layer):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.shape
        g = self.groups
        # RISK_INFO: [张量操作差异] - paddle.transpose 的 perm 参数是 list，而 torch.permute 的 dim 参数是 *args，虽然此处逻辑等价，但属于不同框架的实现差异。
        return x.reshape([N, g, C//g, H, W]).transpose([0, 2, 1, 3, 4]).reshape([N, C, H, W])


class SplitBlock(nn.Layer):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.shape[1] * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Layer):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2D(in_channels, in_channels,
                               kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        self.conv2 = nn.Conv2D(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(in_channels)
        self.conv3 = nn.Conv2D(in_channels, in_channels,
                               kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = paddle.concat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2D(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        self.conv2 = nn.Conv2D(in_channels, mid_channels,
                               kernel_size=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(mid_channels)
        # right
        self.conv3 = nn.Conv2D(in_channels, mid_channels,
                               kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(mid_channels)
        self.conv4 = nn.Conv2D(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(mid_channels)
        self.conv5 = nn.Conv2D(mid_channels, mid_channels,
                               kernel_size=1, bias_attr=False)
        self.bn5 = nn.BatchNorm2D(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = paddle.concat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Layer):
    def __init__(self, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2D(3, 24, kernel_size=3,
                               stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2D(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.avg_pool2d 与 torch.nn.functional.avg_pool2d 在处理边界和填充的默认行为上可能存在细微差异。
        out = F.avg_pool2d(out, 4)
        # RISK_INFO: [张量操作差异] - paddle.reshape 与 torch.view 行为类似，但 torch.view 对内存连续性有要求，而 paddle.reshape 无此限制，在某些边缘情况下行为可能不等价。
        out = out.reshape([out.shape[0], -1])
        out = self.linear(out)
        return out


configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (244, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def test():
    # CRITICAL_ERROR: [预设内容不一致] - 'net_size' 在原作中为 1，此处为 0.5。
    # net = ShuffleNetV2(net_size=0.5) # 不一致的实现
    # # 原架构中的对应代码
    # # net = ShuffleNetV2(net_size=1)
    raise NotImplementedError("'net_size' 在原作中为 1，此处为 0.5。")
    # RISK_INFO: [随机性处理差异] - paddle.randn 与 torch.randn 的随机数生成器不同，即使种子相同，结果也无法保证一致，可能影响复现性。
    x = paddle.randn([3, 3, 32, 32])
    y = net(x)
    print(y.shape)


# test()