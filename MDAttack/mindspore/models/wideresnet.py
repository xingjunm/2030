import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal, One, Zero


class BasicBlock(nn.Cell):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               pad_mode='pad',
                               has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes,
                               out_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               pad_mode='pad',
                               has_bias=False)
        self.droprate = dropRate
        self.dropout = nn.Dropout(keep_prob=1.0 - dropRate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            pad_mode='valid',
            has_bias=False) or None

    def construct(self, x):
        if not self.equalInOut:
            x_processed = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x_processed)))
        if self.droprate > 0:
            out = self.dropout(out)
        out = self.conv2(out)
        return ops.add(x if self.equalInOut else self.convShortcut(x_processed), out)


class NetworkBlock(nn.Cell):
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
        return nn.SequentialCell(layers)

    def construct(self, x):
        return self.layer(x)


class WideResNet(nn.Cell):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3,
                               nChannels[0],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               pad_mode='pad',
                               has_bias=False)
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
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Dense(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.latent = False
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                sigma = math.sqrt(2. / n)
                # Generate random values directly using MindSpore ops
                shape = cell.weight.data.shape
                weight_data = ms.Tensor(np.random.normal(0.0, sigma, shape), dtype=ms.float32)
                cell.weight.set_data(weight_data)
            elif isinstance(cell, nn.BatchNorm2d):
                # Set gamma to ones and beta to zeros
                cell.gamma.set_data(ops.ones(cell.gamma.data.shape, dtype=ms.float32))
                cell.beta.set_data(ops.zeros(cell.beta.data.shape, dtype=ms.float32))
            elif isinstance(cell, nn.Dense):
                # Set bias to zeros
                cell.bias.set_data(ops.zeros(cell.bias.data.shape, dtype=ms.float32))

    def construct(self, x, return_features=[]):
        features = {}
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        # Expose features in the final residual blocks
        sequential = self.block3.layer
        features_list = []
        for s in sequential:
            out = s(out)
            features_list.append(out)
        out = self.relu(self.bn1(out))
        # MindSpore uses different pooling API
        out = ops.avg_pool2d(out, kernel_size=8, stride=8)
        pooled = out.view(-1, self.nChannels)
        output = self.fc(pooled)
        if self.latent:
            features_list += [pooled, output]
            return features_list
        return output