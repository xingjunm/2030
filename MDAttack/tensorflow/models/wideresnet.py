import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BasicBlock(keras.Model):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.droprate = dropRate
        self.dropout = layers.Dropout(rate=dropRate)
        self.equalInOut = (in_planes == out_planes)
        
        if not self.equalInOut:
            self.convShortcut = layers.Conv2D(
                filters=out_planes,
                kernel_size=1,
                strides=stride,
                padding='valid',
                use_bias=False
            )
        else:
            self.convShortcut = None

    def call(self, x, training=None):
        if not self.equalInOut:
            x_relu = tf.nn.relu(self.bn1(x, training=training))
            out = self.conv1(x_relu)
        else:
            out = tf.nn.relu(self.bn1(x, training=training))
            out = self.conv1(out)
        
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        
        if self.droprate > 0:
            out = self.dropout(out, training=training)
        
        out = self.conv2(out)
        
        if self.equalInOut:
            return x + out
        else:
            return self.convShortcut(x_relu) + out


class NetworkBlock(keras.Model):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers_list = []
        for i in range(int(nb_layers)):
            if i == 0:
                layers_list.append(
                    block(in_planes, out_planes, stride, dropRate)
                )
            else:
                layers_list.append(
                    block(out_planes, out_planes, 1, dropRate)
                )
        return layers_list

    def call(self, x, training=None):
        for layer in self.layer:
            x = layer(x, training=training)
        return x


class WideResNet(keras.Model):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = layers.Conv2D(
            filters=nChannels[0],
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self._conv_init
        )
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # global average pooling and classifier
        self.bn1 = layers.BatchNormalization()
        self.fc = layers.Dense(num_classes, kernel_initializer='zeros')
        self.nChannels = nChannels[3]
        self.latent = False

    def _conv_init(self, shape, dtype=None):
        """Initialize conv weights with He normal initialization"""
        fan_in = shape[0] * shape[1] * shape[2]
        stddev = math.sqrt(2. / fan_in)
        return tf.random.normal(shape, mean=0.0, stddev=stddev, dtype=dtype)

    def build(self, input_shape):
        """Build the model by running a dummy input through it"""
        super(WideResNet, self).build(input_shape)
        
        # Initialize BatchNorm weights after building
        for layer in self.layers:
            if isinstance(layer, layers.BatchNormalization):
                if hasattr(layer, 'gamma') and layer.gamma is not None:
                    layer.gamma.assign(tf.ones_like(layer.gamma))
                if hasattr(layer, 'beta') and layer.beta is not None:
                    layer.beta.assign(tf.zeros_like(layer.beta))
            elif isinstance(layer, NetworkBlock):
                for block in layer.layer:
                    for sublayer in [block.bn1, block.bn2]:
                        if hasattr(sublayer, 'gamma') and sublayer.gamma is not None:
                            sublayer.gamma.assign(tf.ones_like(sublayer.gamma))
                        if hasattr(sublayer, 'beta') and sublayer.beta is not None:
                            sublayer.beta.assign(tf.zeros_like(sublayer.beta))

    def call(self, x, training=None, return_features=None):
        """Forward pass through the network"""
        if return_features is None:
            return_features = []
            
        features = {}
        out = self.conv1(x)
        out = self.block1(out, training=training)
        out = self.block2(out, training=training)
        
        # Expose features in the final residual blocks
        features_list = []
        for block in self.block3.layer:
            out = block(out, training=training)
            features_list.append(out)
        
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = tf.nn.avg_pool2d(out, ksize=8, strides=8, padding='VALID')
        pooled = tf.reshape(out, [-1, self.nChannels])
        output = self.fc(pooled)
        
        if self.latent:
            features_list += [pooled, output]
            return features_list
        return output