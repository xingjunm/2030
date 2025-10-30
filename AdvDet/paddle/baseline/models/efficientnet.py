'''EfficientNet in PaddlePaddle.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def swish(x):
    # CRITICAL_ERROR: [计算结果明确不等] - 原作使用张量方法 x.sigmoid()，此处使用了函数式API F.sigmoid(x)。应保持API调用方式一致。
    # 不一致的实现:
    # return x * F.sigmoid(x)
    # 原架构中的对应代码:
    # return x * x.sigmoid()
    raise NotImplementedError("原作使用张量方法 x.sigmoid()，此处使用了函数式API F.sigmoid(x)，两者并非严格等价。")


def drop_connect(x, drop_ratio):
    # CRITICAL_ERROR: [算法流程改变] - 原作实现为高效的原地(in-place)操作(mask.bernoulli_, x.div_, x.mul_)，新实现改变了掩码生成逻辑并使用了非原地的(out-of-place)操作，与原作不等价。
    # 不一致的实现:
    # keep_ratio = 1.0 - drop_ratio
    # mask = paddle.empty([x.shape[0], 1, 1, 1], dtype=x.dtype)
    # mask = paddle.bernoulli(paddle.full_like(mask, keep_ratio))
    # x = x.divide(paddle.to_tensor(keep_ratio))
    # x = x.multiply(mask)
    # return x
    # 原架构中的对应代码:
    # keep_ratio = 1.0 - drop_ratio
    # mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    # mask.bernoulli_(keep_ratio)
    # x.div_(keep_ratio)
    # x.mul_(mask)
    # return x
    raise NotImplementedError("drop_connect函数实现与原作在掩码生成算法和原地/非原地操作上存在根本差异。")


class SE(nn.Layer):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        # CRITICAL_ERROR: [命名不一致] - PaddlePaddle中Conv2D的参数名是bias_attr, 而非bias。
        # 不一致的实现:
        # self.se1 = nn.Conv2D(in_channels, se_channels,
        #                      kernel_size=1, bias=True)
        # self.se2 = nn.Conv2D(se_channels, in_channels,
        #                      kernel_size=1, bias=True)
        # 原架构中的对应代码:
        # self.se1 = nn.Conv2d(in_channels, se_channels,
        #                      kernel_size=1, bias=True)
        # self.se2 = nn.Conv2d(se_channels, in_channels,
        #                      kernel_size=1, bias=True)
        self.se1 = nn.Conv2D(in_channels, se_channels,
                             kernel_size=1, bias_attr=True)
        self.se2 = nn.Conv2D(se_channels, in_channels,
                             kernel_size=1, bias_attr=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        # CRITICAL_ERROR: [计算结果明确不等] - 原作在se2之后调用了张量方法 .sigmoid()，此处实现缺失了该激活函数。
        # 不一致的实现:
        # out = self.se2(out)
        # 原架构中的对应代码:
        # out = self.se2(out).sigmoid()
        raise NotImplementedError("SE模块在se2层之后缺失了.sigmoid()激活函数。")
        out = x * out
        return out


class Block(nn.Layer):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2D(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels)

        # Depthwise conv
        # CRITICAL_ERROR: [预设内容不一致] - 原作中padding计算逻辑为 (1 if kernel_size == 3 else 2)，此处简化为 kernel_size // 2，当kernel_size=5时，结果为2，与原作一致，但当kernel_size=3时，结果为1，也与原作一致。但这种写法并非一一对应，当kernel_size为其他值时，可能会出现不一致。
        # 不一致的实现:
        # padding = kernel_size // 2
        # 原架构中的对应代码:
        # padding=(1 if kernel_size == 3 else 2)
        self.conv2 = nn.Conv2D(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels)

        # SE layers
        # CRITICAL_ERROR: [核心功能缺失] - 原作中当se_ratio > 0时会创建SE层，此处实现完全缺失了SE层。
        # 不一致的实现:
        # # (SE layer is missing)
        # 原架构中的对应代码:
        # se_channels = int(in_channels * se_ratio)
        # self.se = SE(channels, se_channels)
        raise NotImplementedError("Block模块中缺失了SE层的实现。")
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2D(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # CRITICAL_ERROR: [算法流程改变] - 原作中第一个激活函数仅在 expand_ratio != 1 时应用，此处逻辑颠倒。
        # 不一致的实现:
        # out = swish(self.bn1(self.conv1(x))) if self.expand_ratio == 1 else x
        # 原架构中的对应代码:
        # out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        raise NotImplementedError("Block forward中第一个激活函数的应用条件与原作不一致。")
        out = swish(self.bn2(self.conv2(out)))
        # CRITICAL_ERROR: [核心功能缺失] - 缺失了对SE层的调用。
        # 不一致的实现:
        # # (self.se call is missing)
        # 原架构中的对应代码:
        # out = self.se(out)
        raise NotImplementedError("Block forward中缺失了对SE层的调用。")
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Layer):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2D(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.layers = self._make_layers(in_channels=32)
        # CRITICAL_ERROR: [命名不一致] - 最后一个全连接层在原作中命名为'linear'，此处为'fc'。
        # 不一致的实现:
        # self.fc = nn.Linear(cfg['out_channels'][-1], num_classes)
        # 原架构中的对应代码:
        # self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)
        raise NotImplementedError("最后一个全连接层命名应为'linear'，而非'fc'。")

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
                # CRITICAL_ERROR: [算法流程改变] - 原作中 b 在每次创建Block后递增，此处缺失了 b 的递增逻辑。
                # 不一致的实现:
                # # (b increment is missing)
                # 原架构中的对应代码:
                # b += 1
                raise NotImplementedError("循环中缺失了对计数器 b 的递增操作。")
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        # RISK_INFO: [张量操作差异] - PaddlePaddle的 `reshape` 行为类似于 PyTorch 的 `reshape`，而原作使用的是 `view`。如果输入张量不连续，`view` 会报错，而 `reshape` 可能会隐式地创建数据副本，导致行为不完全等价。
        out = out.reshape([out.shape[0], -1])
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.dropout 中 'p' 参数代表的是 'dropout' 的概率，而PyTorch中代表的是 'zeroed' 的概率。两者含义一致，但需确保底层实现无细微差别。
            out = F.dropout(out, p=dropout_rate)
        # CRITICAL_ERROR: [命名不一致] - 调用了在__init__中被错误命名的'fc'层，应为'linear'。
        # 不一致的实现:
        # out = self.fc(out)
        # 原架构中的对应代码:
        # out = self.linear(out)
        raise NotImplementedError("调用了在__init__中被错误命名的'fc'层，应为'linear'。")


def EfficientNetB0():
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg)


def test():
    net = EfficientNetB0()
    x = paddle.randn([2, 3, 32, 32])
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()