'''MobileNetV2 in PaddlePaddle.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Block(nn.Layer):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        # CRITICAL_ERROR: [命名不一致] - 参数 'bias_attr' 在原作中为 'bias'。
        # 不一致的实现:
        # self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        # 原架构中的对应代码:
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        raise NotImplementedError("参数 'bias_attr' 在原作中为 'bias'。")
        # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
        self.bn1 = nn.BatchNorm2D(planes)
        # CRITICAL_ERROR: [命名不一致] - 参数 'bias_attr' 在原作中为 'bias'。
        # 不一致的实现:
        # self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias_attr=False)
        # 原架构中的对应代码:
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        raise NotImplementedError("参数 'bias_attr' 在原作中为 'bias'。")
        # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
        self.bn2 = nn.BatchNorm2D(planes)
        # CRITICAL_ERROR: [命名不一致] - 参数 'bias_attr' 在原作中为 'bias'。
        # 不一致的实现:
        # self.conv3 = nn.Conv2D(planes, out_planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        # 原架构中的对应代码:
        # self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        raise NotImplementedError("参数 'bias_attr' 在原作中为 'bias'。")
        # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
        self.bn3 = nn.BatchNorm2D(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            # CRITICAL_ERROR: [算法流程改变] - shortcut的构建逻辑因其内部组件存在[命名不一致]('bias_attr')而中断。
            # 不一致的实现:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias_attr=False),
            #     # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
            #     nn.BatchNorm2D(out_planes),
            # )
            # 原架构中的对应代码:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            #     nn.BatchNorm2d(out_planes),
            # )
            raise NotImplementedError("shortcut的构建逻辑因其内部组件存在[命名不一致]('bias_attr')而中断。")

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同。
        out = F.relu(self.bn1(self.conv1(x)))
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同。
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Layer):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # CRITICAL_ERROR: [命名不一致] - 参数 'bias_attr' 在原作中为 'bias'。
        # 不一致的实现:
        # self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1, bias_attr=False)
        # 原架构中的对应代码:
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        raise NotImplementedError("参数 'bias_attr' 在原作中为 'bias'。")
        # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
        self.bn1 = nn.BatchNorm2D(32)
        self.layers = self._make_layers(in_planes=32)
        # CRITICAL_ERROR: [命名不一致] - 参数 'bias_attr' 在原作中为 'bias'。
        # 不一致的实现:
        # self.conv2 = nn.Conv2D(320, 1280, kernel_size=1, stride=1, padding=0, bias_attr=False)
        # 原架构中的对应代码:
        # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        raise NotImplementedError("参数 'bias_attr' 在原作中为 'bias'。")
        # RISK_INFO: [API 行为不等价] - BatchNorm实现可能在计算running mean/var时存在细微差异，或在eval模式下行为不一致。
        self.bn2 = nn.BatchNorm2D(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同。
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle的 F.relu 可能与PyTorch在处理某些边缘情况(如NaN)时行为不同。
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # RISK_INFO: [API 行为不等价] - F.avg_pool2d 在不同框架间可能存在实现差异，尤其是在处理边界或取整时。
        out = F.avg_pool2d(out, 4)
        # CRITICAL_ERROR: [算法流程改变] - 原作使用 'view' 方法进行张量变形，新代码使用了 'reshape'，两者在处理非连续内存的张量时行为不同。同时原作使用 '.size()' 方法，新代码使用 '.shape' 属性。
        # 不一致的实现:
        # out = out.reshape([out.shape[0], -1])
        # 原架构中的对应代码:
        # out = out.view(out.size(0), -1)
        raise NotImplementedError("原作使用 'view' 方法进行张量变形，新代码使用了 'reshape'，两者在处理非连续内存的张量时行为不同。同时原作使用 '.size()' 方法，新代码使用 '.shape' 属性。")
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    # RISK_INFO: [API 行为不等价] - paddle.randn 使用列表 `[shape]` 作为参数，而 torch.randn 使用可变参数 `*shape`。
    x = paddle.randn([2,3,32,32])
    y = net(x)
    # CRITICAL_ERROR: [命名不一致] - 原作使用 'y.size()' 方法获取张量大小，而新代码使用了 '.shape' 属性。
    # 不一致的实现:
    # print(y.shape)
    # 原架构中的对应代码:
    # print(y.size())
    raise NotImplementedError("原作使用 'y.size()' 方法获取张量大小，而新代码使用了 '.shape' 属性。")

# test()