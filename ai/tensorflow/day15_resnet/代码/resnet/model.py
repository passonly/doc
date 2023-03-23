from tensorflow import keras
import tensorflow as tf
import numpy as np


class BasicBlock(keras.layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(out_channel,
                                         kernel_size=3,
                                         strides=strides,
                                         padding='SAME',
                                         use_bias=False) # 一般如果使用了BN层, 就不用bias
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(out_channel,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = keras.layers.ReLU()
        self.add = keras.layers.Add()
        self.downsameple = downsample

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(keras.layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(out_channel,
                                         kernel_size=1,
                                         strides=1,
                                         use_bias=False) # 一般如果使用了BN层, 就不用bias
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(out_channel,
                                         kernel_size=3,
                                         strides=strides,
                                         padding='SAME',
                                         use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = keras.layers.Conv2D(out_channel * self.expansion,
                                         kernel_size=1,
                                         strides=1,
                                         use_bias=False)
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = keras.layers.ReLU()
        self.add = keras.layers.Add()
        self.downsameple = downsample

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


def make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None

    if strides != 1 or in_channel != channel * block.expansion:
        downsample = keras.Sequential([
                keras.layers.Conv2D(channel * block.expansion, kernel_size=1,
                                    strides=strides,
                                    use_bias=False, name='conv1'),
                keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                                name='BatchNorm')
        ], name='shortcut')

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides,
                            name='unit_1'))
    for index in range(1, block_num):
        layers_list.append(block(channel, strides=strides,
                                name='unit_' + str(index + 1)))
    return keras.Sequential(layers_list, name=name)

