import tensorflow as tf
import tensorflow.keras.layers as layers


class DarkNet(tf.keras.Model):
    def darknetConv2d(self, kernel_size=None, filters=-1, use_bias=False, pre_lay_ch_cnt=-1):
        if kernel_size is None:
            kernel_size = [3, 3]
        # 记录每个卷积核参数的shape
        self.weight_shapes.append((kernel_size[0], kernel_size[1], pre_lay_ch_cnt, filters))
        return layers.Conv2D(
            kernel_size=kernel_size,
            filters=filters,
            padding='same',
            strides=1,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        )

    def __init__(self):
        super().__init__()

        self.weight_shapes = []

        self.conv1 = self.darknetConv2d(
            filters=32,
            pre_lay_ch_cnt=3,
        )

        self.bn1 = layers.BatchNormalization()
        self.activation1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.pool1 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv2 = self.darknetConv2d(
            filters=64,
            pre_lay_ch_cnt=32
        )
        self.bn2 = layers.BatchNormalization()
        self.activation2 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.pool2 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv3 = self.darknetConv2d(
            filters=128,
            pre_lay_ch_cnt=64
        )
        self.bn3 = layers.BatchNormalization()
        self.activation3 = tf.keras.layers.LeakyReLU(alpha=0.1)

        '''
           构建瓶颈层，减少计算量，参考https://www.bilibili.com/video/BV1F4411y7o7?p=17
           n*n*128 -> same,3,3 -> n*n*128
           计算量 = n*n*128*3*3*128 = 3*3*128*128*n*n = 18*(128*64*n*n)
           n*n*128 -> 1,1 -> n*n*64 -> same,3,3 -> n*n*128
           计算量 = n*n*64*1*1*128 + n*n*128*3*3*64 = 10*(128*64*n*n)
           10*(128*64*n*n) / 18*(128*64*n*n) = 0.556
           缩小到原来一半的计算量
        '''

        self.conv4 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=64,
            pre_lay_ch_cnt=128
        )
        self.bn4 = layers.BatchNormalization()
        self.activation4 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv5 = self.darknetConv2d(
            filters=128,
            pre_lay_ch_cnt=64
        )
        self.bn5 = layers.BatchNormalization()
        self.activation5 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.pool5 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv6 = self.darknetConv2d(
            filters=256,
            pre_lay_ch_cnt=128
        )
        self.bn6 = layers.BatchNormalization()
        self.activation6 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv7 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=128,
            pre_lay_ch_cnt=256
        )
        self.bn7 = layers.BatchNormalization()
        self.activation7 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv8 = self.darknetConv2d(
            filters=256,
            pre_lay_ch_cnt=128
        )
        self.bn8 = layers.BatchNormalization()
        self.activation8 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.pool8 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv9 = self.darknetConv2d(
            filters=512,
            pre_lay_ch_cnt=256
        )
        self.bn9 = layers.BatchNormalization()
        self.activation9 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv10 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=256,
            pre_lay_ch_cnt=512
        )
        self.bn10 = layers.BatchNormalization()
        self.activation10 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv11 = self.darknetConv2d(
            filters=512,
            pre_lay_ch_cnt=256
        )
        self.bn11 = layers.BatchNormalization()
        self.activation11 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv12 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=256,
            pre_lay_ch_cnt=512
        )
        self.bn12 = layers.BatchNormalization()
        self.activation12 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv13 = self.darknetConv2d(
            filters=512,
            pre_lay_ch_cnt=256
        )
        self.bn13 = layers.BatchNormalization()
        self.activation13 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.pool13 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv14 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=512
        )
        self.bn14 = layers.BatchNormalization()
        self.activation14 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv15 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=512,
            pre_lay_ch_cnt=1024
        )
        self.bn15 = layers.BatchNormalization()
        self.activation15 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv16 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=512
        )
        self.bn16 = layers.BatchNormalization()
        self.activation16 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv17 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=512,
            pre_lay_ch_cnt=1024
        )
        self.bn17 = layers.BatchNormalization()
        self.activation17 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv18 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=512
        )
        self.bn18 = layers.BatchNormalization()
        self.activation18 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv19 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=1024
        )
        self.bn19 = layers.BatchNormalization()
        self.activation19 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv20 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=1024
        )
        self.bn20 = layers.BatchNormalization()
        self.activation20 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv21 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=64,
            pre_lay_ch_cnt=512
        )
        self.bn21 = layers.BatchNormalization()
        self.activation21 = layers.LeakyReLU(alpha=0.1)
        self.lambda1 = tf.keras.layers.Lambda(lambda x: tf.nn.space_to_depth(x, block_size=2, data_format="NHWC"),
                                              name='space_to_depth_b2')
        self.concatenate1 = layers.Concatenate()
        self.conv22 = self.darknetConv2d(
            filters=1024,
            pre_lay_ch_cnt=1280
        )
        self.bn22 = layers.BatchNormalization()
        self.activation22 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv23 = self.darknetConv2d(
            kernel_size=[1, 1],
            filters=425,
            use_bias=True,
            pre_lay_ch_cnt=1024,
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.activation6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.activation7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.activation8(x)
        x = self.pool8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.activation9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.activation10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.activation11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.activation12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        leaky_re_lu_13 = x = self.activation13(x)
        x = self.pool13(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = self.activation14(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = self.activation15(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = self.activation16(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = self.activation17(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = self.activation18(x)

        x = self.conv19(x)
        x = self.bn19(x)
        x = self.activation19(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = self.activation20(x)
        # 跳跃连接
        y = self.conv21(leaky_re_lu_13)
        y = self.bn21(y)
        y = self.activation21(y)

        y = self.lambda1(y)
        x = self.concatenate1([y, x])

        x = self.conv22(x)
        x = self.bn22(x)
        x = self.activation22(x)

        x = self.conv23(x)
        return x
