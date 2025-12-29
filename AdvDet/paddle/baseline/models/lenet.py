'''LeNet in PaddlePaddle.'''
import paddle.nn as nn
import paddle.nn.functional as F

def __Conv2d(*args, **kwargs):
    """使用PaddlePaddle的Conv2D替代PyTorch的Conv2d，因为框架API命名不同"""
    return nn.Conv2D(*args, **kwargs)

def __view(tensor, *shape):
    """使用PaddlePaddle的reshape替代PyTorch的view，因为框架API命名不同"""
    return tensor.reshape(shape)

class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = __Conv2d(3, 6, 5)
        self.conv2 = __Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理NaN等边缘情况时行为不同。
        out = F.relu(self.conv1(x))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.max_pool2d 的默认参数(如stride, padding)和底层实现可能与PyTorch版本存在差异。在PyTorch和Paddle中，当stride未指定时，其默认值均等于kernel_size，故此处行为一致，但仍需注意潜在风险。
        out = F.max_pool2d(out, 2)
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理NaN等边缘情况时行为不同。
        out = F.relu(self.conv2(out))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.max_pool2d 的默认参数(如stride, padding)和底层实现可能与PyTorch版本存在差异。在PyTorch和Paddle中，当stride未指定时，其默认值均等于kernel_size，故此处行为一致，但仍需注意潜在风险。
        out = F.max_pool2d(out, 2)
        # RISK_INFO: [张量操作差异] - paddle的reshape与torch的view功能上等价，但torch的view要求张量在内存上是连续的，而reshape无此要求。这是一个潜在的细微差异。
        out = __view(out, out.shape[0], -1)
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理NaN等边缘情况时行为不同。
        out = F.relu(self.fc1(out))
        # RISK_INFO: [API 行为不等价] - paddle.nn.functional.relu 可能与 torch.nn.functional.relu 在处理NaN等边缘情况时行为不同。
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out