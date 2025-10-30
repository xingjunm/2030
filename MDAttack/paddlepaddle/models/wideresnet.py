import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BasicBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes, momentum=0.95)  # Exemption #11: PyTorch 0.05 -> PaddlePaddle 0.95
        self.relu1 = nn.ReLU()  # Exemption #10: PaddlePaddle nn.ReLU doesn't support inplace
        self.conv1 = nn.Conv2D(in_planes,
                               out_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_planes, momentum=0.95)  # Exemption #11: PyTorch 0.05 -> PaddlePaddle 0.95
        self.relu2 = nn.ReLU()  # Exemption #10: PaddlePaddle nn.ReLU doesn't support inplace
        self.conv2 = nn.Conv2D(out_planes,
                               out_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias_attr=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return paddle.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Layer):
    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Layer):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2D(3,
                               nChannels[0],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias_attr=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2D(nChannels[3], momentum=0.95)  # Exemption #11: PyTorch 0.05 -> PaddlePaddle 0.95
        self.relu = nn.ReLU()  # Exemption #10: PaddlePaddle nn.ReLU doesn't support inplace
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.latent = False

        # Initialize weights
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                # PaddlePaddle uses different initialization API
                normal_init = paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. / n))
                normal_init(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                # PaddlePaddle uses different initialization API
                constant_init_1 = paddle.nn.initializer.Constant(value=1.0)
                constant_init_0 = paddle.nn.initializer.Constant(value=0.0)
                constant_init_1(m.weight)
                constant_init_0(m.bias)
            elif isinstance(m, nn.Linear):
                # PaddlePaddle uses different initialization API
                constant_init_0 = paddle.nn.initializer.Constant(value=0.0)
                constant_init_0(m.bias)

    def forward(self, x, return_features=[]):
        features = {}
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        # Expose features in the final residual blocks
        sequential = self.block3.layer
        features = []
        for s in sequential:
            out = s(out)
            features.append(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        pooled = out.reshape([-1, self.nChannels])  # PaddlePaddle uses reshape instead of view
        output = self.fc(pooled)
        if self.latent:
            features += [pooled, output]
            return features
        return output