import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import concatenate


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size   = 3,
                 strides       = 1,
                 padding       = 'SAME',
                 dilation_rate = 1,
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.strides       = strides
        self.padding       = padding
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.conv = Conv2D(filters=self.filters, 
                           kernel_size=self.kernel_size, 
                           strides=self.strides,
                           padding=self.padding,
                           dilation_rate=self.dilation_rate)
        self.norm_layer = BatchNormalization()
        self.activation = Activation('relu')

    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.norm_layer(x, training=training)
        x = self.activation(x)
        return x


class RSU7(tf.keras.layers.Layer):
    def __init__(self, 
                 filters=[12, 3],
                 **kwargs):
        super(RSU7, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        f0, f1 = self.filters
        self.rebnconvin = ConvolutionBlock(filters=f1, dilation_rate=1)

        self.rebnconv1  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool1      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv2  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool2      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv3  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool3      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv4  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool4      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv5  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool5      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv6  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv7  = ConvolutionBlock(filters=f0, dilation_rate=2)

        self.rebnconv6d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv6u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv5d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv5u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv4d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv4u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv3d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv2d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv1d = ConvolutionBlock(filters=f1, dilation_rate=1)

    def call(self, inputs, training=False):
        inputs = self.rebnconvin(inputs, training=training)

        P1     = self.rebnconv1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.rebnconv2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.rebnconv3(P_down, training=training)
        P_down = self.pool3(P3)

        P4     = self.rebnconv4(P_down, training=training)
        P_down = self.pool4(P4)

        P5     = self.rebnconv5(P_down, training=training)
        P_down = self.pool5(P5)

        P6     = self.rebnconv6(P_down, training=training)

        P7     = self.rebnconv7(P6, training=training)

        P6_up  = self.rebnconv6d(concatenate([P7, P6], axis=-1), training=training)
        P6_up  = self.rebnconv6u(P6_up)

        P5_up  = self.rebnconv5d(concatenate([P6_up, P5], axis=-1), training=training)
        P5_up  = self.rebnconv5u(P5_up)

        P4_up  = self.rebnconv4d(concatenate([P5_up, P4], axis=-1), training=training)
        P4_up  = self.rebnconv4u(P4_up)

        P3_up  = self.rebnconv3d(concatenate([P4_up, P3], axis=-1), training=training)
        P3_up  = self.rebnconv3u(P3_up)

        P2_up  = self.rebnconv2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.rebnconv2u(P2_up)

        P1_up  = self.rebnconv1d(concatenate([P2_up, P1], axis=-1), training=training)
        return P1_up + inputs


class RSU6(tf.keras.layers.Layer):
    def __init__(self, 
                 filters=[12, 3],
                 **kwargs):
        super(RSU6, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        f0, f1 = self.filters
        self.rebnconvin = ConvolutionBlock(filters=f1, dilation_rate=1)

        self.rebnconv1  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool1      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv2  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool2      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv3  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool3      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv4  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool4      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv5  = ConvolutionBlock(filters=f0, dilation_rate=1)

        self.rebnconv6  = ConvolutionBlock(filters=f0, dilation_rate=2)

        self.rebnconv5d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv5u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv4d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv4u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv3d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv2d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv1d = ConvolutionBlock(filters=f1, dilation_rate=1)

    def call(self, inputs, training=False):
        inputs = self.rebnconvin(inputs, training=training)

        P1     = self.rebnconv1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.rebnconv2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.rebnconv3(P_down, training=training)
        P_down = self.pool3(P3)

        P4     = self.rebnconv4(P_down, training=training)
        P_down = self.pool4(P4)

        P5     = self.rebnconv5(P_down, training=training)

        P6     = self.rebnconv6(P5, training=training)

        P5_up  = self.rebnconv5d(concatenate([P6, P5], axis=-1), training=training)
        P5_up  = self.rebnconv5u(P5_up)

        P4_up  = self.rebnconv4d(concatenate([P5_up, P4], axis=-1), training=training)
        P4_up  = self.rebnconv4u(P4_up)

        P3_up  = self.rebnconv3d(concatenate([P4_up, P3], axis=-1), training=training)
        P3_up  = self.rebnconv3u(P3_up)

        P2_up  = self.rebnconv2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.rebnconv2u(P2_up)

        P1_up  = self.rebnconv1d(concatenate([P2_up, P1], axis=-1), training=training)
        return P1_up + inputs


class RSU5(tf.keras.layers.Layer):
    def __init__(self, 
                 filters=[12, 3],
                 **kwargs):
        super(RSU5, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        f0, f1 = self.filters
        self.rebnconvin = ConvolutionBlock(filters=f1, dilation_rate=1)

        self.rebnconv1  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool1      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv2  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool2      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv3  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool3      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv4  = ConvolutionBlock(filters=f0, dilation_rate=1)

        self.rebnconv5  = ConvolutionBlock(filters=f0, dilation_rate=2)

        self.rebnconv4d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv4u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv3d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv2d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv1d = ConvolutionBlock(filters=f1, dilation_rate=1)

    def call(self, inputs, training=False):
        inputs = self.rebnconvin(inputs, training=training)

        P1     = self.rebnconv1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.rebnconv2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.rebnconv3(P_down, training=training)
        P_down = self.pool3(P3)

        P4     = self.rebnconv4(P_down, training=training)

        P5     = self.rebnconv5(P_down, training=training)

        P4_up  = self.rebnconv4d(concatenate([P5, P4], axis=-1), training=training)
        P4_up  = self.rebnconv4u(P4_up)

        P3_up  = self.rebnconv3d(concatenate([P4_up, P3], axis=-1), training=training)
        P3_up  = self.rebnconv3u(P3_up)

        P2_up  = self.rebnconv2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.rebnconv2u(P2_up)

        P1_up  = self.rebnconv1d(concatenate([P2_up, P1], axis=-1), training=training)
        return P1_up + inputs


class RSU4(tf.keras.layers.Layer):
    def __init__(self, 
                 filters=[12, 3],
                 **kwargs):
        super(RSU4, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        f0, f1 = self.filters
        self.rebnconvin = ConvolutionBlock(filters=f1, dilation_rate=1)

        self.rebnconv1  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool1      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv2  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.pool2      = MaxPool2D(pool_size=2, strides=2)

        self.rebnconv3  = ConvolutionBlock(filters=f0, dilation_rate=1)

        self.rebnconv4  = ConvolutionBlock(filters=f0, dilation_rate=2)

        self.rebnconv3d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.rebnconv2d = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.rebnconv1d = ConvolutionBlock(filters=f1, dilation_rate=1)

    def call(self, inputs, training=False):
        inputs = self.rebnconvin(inputs, training=training)

        P1     = self.rebnconv1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.rebnconv2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.rebnconv3(P_down, training=training)

        P4     = self.rebnconv4(P_down, training=training)

        P3_up  = self.rebnconv3d(concatenate([P4, P3], axis=-1), training=training)
        P3_up  = self.rebnconv3u(P3_up)

        P2_up  = self.rebnconv2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.rebnconv2u(P2_up)

        P1_up  = self.rebnconv1d(concatenate([P2_up, P1], axis=-1), training=training)
        return P1_up + inputs


class RSU4F(tf.keras.layers.Layer):
    def __init__(self, 
                 filters=[12, 3],
                 **kwargs):
        super(RSU4F, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        f0, f1 = self.filters
        self.rebnconvin = ConvolutionBlock(filters=f1, dilation_rate=1)

        self.rebnconv1  = ConvolutionBlock(filters=f0, dilation_rate=1)
        self.rebnconv2  = ConvolutionBlock(filters=f0, dilation_rate=2)
        self.rebnconv3  = ConvolutionBlock(filters=f0, dilation_rate=4)
        self.rebnconv4  = ConvolutionBlock(filters=f0, dilation_rate=8)

        self.rebnconv3d = ConvolutionBlock(filters=f0, dilation_rate=4)        
        self.rebnconv2d = ConvolutionBlock(filters=f0, dilation_rate=2)
        self.rebnconv1d = ConvolutionBlock(filters=f1, dilation_rate=1)

    def call(self, inputs, training=False):
        inputs = self.rebnconvin(inputs, training=training)

        P1     = self.rebnconv1(inputs, training=training)
        P2     = self.rebnconv2(P1, training=training)
        P3     = self.rebnconv3(P2, training=training)
        P4     = self.rebnconv4(P3, training=training)

        P3_up  = self.rebnconv3d(concatenate([P4, P3], axis=-1), training=training)
        P2_up  = self.rebnconv2d(concatenate([P3_up, P2], axis=-1), training=training)
        P1_up  = self.rebnconv1d(concatenate([P2_up, P1], axis=-1), training=training)
        return P1_up + inputs


class U2Net(tf.keras.Model):
    def __init__(self, 
                 filters=[16, 32, 64, 128, 256, 512],
                 classes=1,
                 **kwargs):
        super(U2Net, self).__init__(**kwargs)
        self.filters = filters
        self.classes = classes

    def build(self, input_shape):
        f0, f1, f2, f3, f4, f5 = self.filters        
        self.stage1  = RSU7([f1, f2])
        self.pool1   = MaxPool2D(pool_size=2, strides=2)

        self.stage2  = RSU6([f1, f3])
        self.pool2   = MaxPool2D(pool_size=2, strides=2)

        self.stage3  = RSU5([f2, f4])
        self.pool3   = MaxPool2D(pool_size=2, strides=2)

        self.stage4  = RSU4([f3, f5])
        self.pool4   = MaxPool2D(pool_size=2, strides=2)

        self.stage5  = RSU4F([f4, f5])
        self.pool5   = MaxPool2D(pool_size=2, strides=2)

        self.stage6  = RSU4F([f4, f5])
        self.stage6u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        # decoder
        self.stage5d = RSU4F([f4, f5])
        self.stage5u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage4d = RSU4([f3, f4])
        self.stage4u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage3d = RSU5([f2, f3])
        self.stage3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.stage2d = RSU6([f1, f2])
        self.stage2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage1d = RSU7([f0, f2])

        # side
        self.side1   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.activ1  = Activation('sigmoid')

        self.side2   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side2u  = UpSampling2D(size=[2, 2], interpolation="bilinear")
        self.activ2  = Activation('sigmoid')

        self.side3   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side3u  = UpSampling2D(size=[4, 4], interpolation="bilinear")
        self.activ3  = Activation('sigmoid')

        self.side4   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side4u  = UpSampling2D(size=[8, 8], interpolation="bilinear")
        self.activ4  = Activation('sigmoid')

        self.side5   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side5u  = UpSampling2D(size=[16, 16], interpolation="bilinear")
        self.activ5  = Activation('sigmoid')

        self.side6   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side6u  = UpSampling2D(size=[32, 32], interpolation="bilinear")
        self.activ6  = Activation('sigmoid')

        self.outconv = Conv2D(filters=self.classes, kernel_size=(1, 1), strides=(1, 1), padding="SAME")
        self.activ0  = Activation('sigmoid')

    def call(self, inputs, training=False):
        P1     = self.stage1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.stage2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.stage3(P_down, training=training)
        P_down = self.pool3(P3)

        P4     = self.stage4(P_down, training=training)
        P_down = self.pool4(P4)

        P5     = self.stage5(P_down, training=training)
        P_down = self.pool5(P5)

        P6     = self.stage6(P_down, training=training)
        P6_up  = self.stage6u(P6)

        #decoder
        P5     = self.stage5d(concatenate([P6_up, P5], axis=-1), training=training)
        P5_up  = self.stage5u(P5)

        P4     = self.stage4d(concatenate([P5_up, P4], axis=-1), training=training)
        P4_up  = self.stage4u(P4)

        P3     = self.stage3d(concatenate([P4_up, P3], axis=-1), training=training)
        P3_up  = self.stage3u(P3)

        P2     = self.stage2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.stage2u(P2)

        P1_up  = self.stage1d(concatenate([P2_up, P1], axis=-1), training=training)

        #side output
        D1     = self.side1(P1_up, training=training)
        D1     = self.activ1(D1)

        D2     = self.side2(P2, training=training)
        D2     = self.side2u(D2)
        D2     = self.activ2(D2)

        D3     = self.side3(P3, training=training)
        D3     = self.side3u(D3)
        D3     = self.activ3(D3)

        D4     = self.side4(P4, training=training)
        D4     = self.side4u(D4)
        D4     = self.activ4(D4)

        D5     = self.side5(P5, training=training)
        D5     = self.side5u(D5)
        D5     = self.activ5(D5)

        D6     = self.side6(P6, training=training)
        D6     = self.side6u(D6)
        D6     = self.activ6(D6)

        D0     = concatenate([D1, D2, D3, D4, D5, D6], axis=-1)
        D0     = self.outconv(D0, training=training)
        D0     = self.activ0(D0)
        return D0, D1, D2, D3, D4, D5, D6


class U2NetP(tf.keras.Model):
    def __init__(self, 
                 filters=[16, 64],
                 classes=1,
                 **kwargs):
        super(U2NetP, self).__init__(**kwargs)
        self.filters = filters
        self.classes = classes

    def build(self, input_shape):
        f0, f1 = self.filters
        self.stage1  = RSU7([f0, f1])
        self.pool1   = MaxPool2D(pool_size=2, strides=2)

        self.stage2  = RSU6([f0, f1])
        self.pool2   = MaxPool2D(pool_size=2, strides=2)

        self.stage3  = RSU5([f0, f1])
        self.pool3   = MaxPool2D(pool_size=2, strides=2)

        self.stage4  = RSU4([f0, f1])
        self.pool4   = MaxPool2D(pool_size=2, strides=2)

        self.stage5  = RSU4F([f0, f1])
        self.pool5   = MaxPool2D(pool_size=2, strides=2)

        self.stage6  = RSU4F([f0, f1])
        self.stage6u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        # decoder
        self.stage5d = RSU4F([f0, f1])
        self.stage5u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage4d = RSU4([f0, f1])
        self.stage4u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage3d = RSU5([f0, f1])
        self.stage3u = UpSampling2D(size=[2, 2], interpolation="bilinear")
        
        self.stage2d = RSU6([f0, f1])
        self.stage2u = UpSampling2D(size=[2, 2], interpolation="bilinear")

        self.stage1d = RSU7([f0, f1])

        # side
        self.side1   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.activ1  = Activation('sigmoid')

        self.side2   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side2u  = UpSampling2D(size=[2, 2], interpolation="bilinear")
        self.activ2  = Activation('sigmoid')

        self.side3   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side3u  = UpSampling2D(size=[4, 4], interpolation="bilinear")
        self.activ3  = Activation('sigmoid')

        self.side4   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side4u  = UpSampling2D(size=[8, 8], interpolation="bilinear")
        self.activ4  = Activation('sigmoid')

        self.side5   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side5u  = UpSampling2D(size=[16, 16], interpolation="bilinear")
        self.activ5  = Activation('sigmoid')

        self.side6   = Conv2D(filters=self.classes, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.side6u  = UpSampling2D(size=[32, 32], interpolation="bilinear")
        self.activ6  = Activation('sigmoid')

        self.outconv = Conv2D(filters=self.classes, kernel_size=(1, 1), strides=(1, 1), padding="SAME")
        self.activ0  = Activation('sigmoid')

    def call(self, inputs, training=False):
        P1     = self.stage1(inputs, training=training)
        P_down = self.pool1(P1)

        P2     = self.stage2(P_down, training=training)
        P_down = self.pool2(P2)

        P3     = self.stage3(P_down, training=training)
        P_down = self.pool3(P3)

        P4     = self.stage4(P_down, training=training)
        P_down = self.pool4(P4)

        P5     = self.stage5(P_down, training=training)
        P_down = self.pool5(P5)

        P6     = self.stage6(P_down, training=training)
        P6_up  = self.stage6u(P6)

        #decoder
        P5     = self.stage5d(concatenate([P6_up, P5], axis=-1), training=training)
        P5_up  = self.stage5u(P5)

        P4     = self.stage4d(concatenate([P5_up, P4], axis=-1), training=training)
        P4_up  = self.stage4u(P4)

        P3     = self.stage3d(concatenate([P4_up, P3], axis=-1), training=training)
        P3_up  = self.stage3u(P3)

        P2     = self.stage2d(concatenate([P3_up, P2], axis=-1), training=training)
        P2_up  = self.stage2u(P2)

        P1_up  = self.stage1d(concatenate([P2_up, P1], axis=-1), training=training)

        #side output
        D1     = self.side1(P1_up, training=training)
        D1     = self.activ1(D1)

        D2     = self.side2(P2, training=training)
        D2     = self.side2u(D2)
        D2     = self.activ2(D2)

        D3     = self.side3(P3, training=training)
        D3     = self.side3u(D3)
        D3     = self.activ3(D3)

        D4     = self.side4(P4, training=training)
        D4     = self.side4u(D4)
        D4     = self.activ4(D4)

        D5     = self.side5(P5, training=training)
        D5     = self.side5u(D5)
        D5     = self.activ5(D5)

        D6     = self.side6(P6, training=training)
        D6     = self.side6u(D6)
        D6     = self.activ6(D6)

        D0     = concatenate([D1, D2, D3, D4, D5, D6], axis=-1)
        D0     = self.outconv(D0, training=training)
        D0     = self.activ0(D0)
        return D0, D1, D2, D3, D4, D5, D6