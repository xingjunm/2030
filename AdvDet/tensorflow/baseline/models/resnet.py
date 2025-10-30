'''ResNet in TensorFlow.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class BasicBlock(keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3,
                               strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.shortcut = keras.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = keras.Sequential([
                layers.Conv2D(self.expansion*planes,
                          kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training=training)
        out = tf.nn.relu(out)
        return out


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3,
                               strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion *
                               planes, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.shortcut = keras.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = keras.Sequential([
                layers.Conv2D(self.expansion*planes,
                          kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        out += self.shortcut(x, training=training)
        out = tf.nn.relu(out)
        return out


class ResNet(keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = layers.Conv2D(64, kernel_size=3,
                               strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return keras.Sequential(layers_list)

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        out = self.avg_pool(out)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    net.build(input_shape=(None, 32, 32, 3))
    y = net(tf.random.normal((1, 32, 32, 3)))
    print(y.shape)

# test()