'''Dual Path Networks in PaddlePaddle.'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from baseline.models import *


class Bottleneck(nn.Layer):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2D(last_planes, in_planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv2 = nn.Conv2D(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(in_planes)
        self.conv3 = nn.Conv2D(in_planes, out_planes+dense_depth, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2D(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        # RISK_INFO: [张量操作差异] - paddle.concat 的功能与 torch.cat 一致，但需要确认在所有边缘情况下（如空张量）行为是否相同。
        out = paddle.concat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Layer):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            # CRITICAL_ERROR: [计算结果明确不等] - last_planes 的更新逻辑与原作不一致。原作是 (i+1)，新代码是 (i+2)。
            # 不一致的实现:
            # self.last_planes = out_planes + (i+2) * dense_depth
            # 原架构中的对应代码:
            # self.last_planes = out_planes + (i+1) * dense_depth
            raise NotImplementedError("last_planes 的更新逻辑与原作不一致。原作是 (i+1)，新代码是 (i+2)。")
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # RISK_INFO: [API 行为不等价] - PyTorch的 F.avg_pool2d 默认 count_include_pad=True，而 Paddle 的 F.avg_pool2d 默认 exclusive=True (等价于 count_include_pad=False)，在有 padding 的情况下行为可能不同。
        out = F.avg_pool2d(out, 4)
        # CRITICAL_ERROR: [算法流程改变] - 原作使用 .view() 进行张量变形，新代码使用 paddle.flatten() 或 .reshape()。这两个操作在内存处理上不等价，.view() 对张量的连续性有要求。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原作使用 .view() 进行张量变形，新代码使用 paddle.flatten() 或 .reshape()。这两个操作在内存处理上不等价，.view() 对张量的连续性有要求。")
        out = self.linear(out)
        return out


def DPN26():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    # CRITICAL_ERROR: [预设内容不一致] - DPN92 配置中的 'dense_depth' 与原作不符。
    # 不一致的实现:
    # cfg = {
    #     'in_planes': (96,192,384,768),
    #     'out_planes': (256,512,1024,2048),
    #     'num_blocks': (3,4,20,3),
    #     'dense_depth': (16,32,24,128)
    # }
    # 原架构中的对应代码:
    # cfg = {
    #     'in_planes': (96,192,384,768),
    #     'out_planes': (256,512,1024,2048),
    #     'num_blocks': (3,4,20,3),
    #     'dense_depth': (16,32,24,128)
    # }
    raise NotImplementedError("DPN92 配置中的 'dense_depth' 与原作不符。")
    return DPN(cfg)


def test():
    net = DPN92()
    # RISK_INFO: [随机性处理差异] - paddle.randn 和 torch.randn 会产生不同的随机数，除非为两个框架设置相同的随机种子，否则无法复现结果。
    x = paddle.randn([1,3,32,32])
    y = net(x)
    print(y)

# test()