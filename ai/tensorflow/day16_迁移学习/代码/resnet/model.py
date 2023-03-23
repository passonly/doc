from tensorflow import keras


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
        self.downsample = downsample

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


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
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
        layers_list.append(block(channel,
                                name='unit_' + str(index + 1)))
    return keras.Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000,
            include_top=True):
    input_image = keras.layers.Input(shape=(im_height, im_width, 3),
                                     dtype='float32')
    x = keras.layers.Conv2D(64, kernel_size=7,
                            strides=2,
                            padding='SAME',
                            name='covn1',
                            use_bias=False)(input_image)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name='block1')(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], name='block2', strides=2)(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], name='block3', strides=2)(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], name='block4', strides=2)(x)

    if include_top:
        # 等于avg pool + flatten
        x = keras.layers.GlobalAvgPool2D()(x)
        x = keras.layers.Dense(num_classes)(x)
        predict = keras.layers.Softmax()(x)
    else:
        predict = x

    model = keras.models.Model(inputs=input_image, outputs=predict)
    return model


def resnet34(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(BasicBlock,
                   [3, 4, 6, 3],
                   im_width,
                   im_height,
                   num_classes,
                   include_top)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck,
                   [3, 4, 6, 3],
                   im_width,
                   im_height,
                   num_classes,
                   include_top)
