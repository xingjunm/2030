'''VGG11/13/16/19 in PaddlePaddle.'''
import paddle
import paddle.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Layer):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        # RISK_INFO: [张量操作差异] - paddle.reshape 与 torch.view 功能相似，但在处理非连续张量时行为不同。torch.view 要求张量是连续的，而 paddle.reshape 可能会在需要时隐式地创建副本，这可能引入性能或内存开销。
        out = out.reshape([out.shape[0], -1])
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
            else:
                # CRITICAL_ERROR: [核心功能缺失] - 原架构中使用了 inplace=True 的 ReLU，新架构的 nn.ReLU() 不支持 inplace 操作，导致行为不一致。
                # layers += [nn.Conv2D(in_channels, x, kernel_size=3, padding=1),
                #            nn.BatchNorm2D(x),
                #            nn.ReLU()] # 不一致的实现
                # 原架构中的对应代码: layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                raise NotImplementedError("原架构中使用了 inplace=True 的 ReLU，新架构的 nn.ReLU() 不支持 inplace 操作，导致行为不一致。")
                in_channels = x
        layers += [nn.AvgPool2D(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = paddle.randn([2,3,32,32])
    y = net(x)
    print(y.shape)

# test()