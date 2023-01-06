import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Identity
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size   = 3,
                 strides       = 1,
                 padding       = 'SAME',
                 dilation_rate = 1,
                 activation    = 'relu',
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.strides       = strides
        self.padding       = padding
        self.dilation_rate = dilation_rate
        self.activation    = activation
        
    def build(self, input_shape):
        self.conv = Conv2D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding=self.padding)
        self.norm_layer = BatchNormalization()
        self.acti_layer = Activation(self.activation) if self.activation else Identity()

    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.norm_layer(x, training=training)
        x = self.acti_layer(x)
        return x
    
    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.block0 = ConvolutionBlock(self.filters, 3, 1, "SAME", 1, "relu")
        self.block1 = ConvolutionBlock(self.filters, 3, 1, "SAME", 1, None)
        
        self.residual = ConvolutionBlock(self.filters, 1, 1, "SAME", 1, None)
        self.final_activation = Activation('relu')
        
    def call(self, inputs, training=False):
        x = self.block0(inputs, training=training)
        x = self.block1(x, training=training)
        y = self.residual(inputs, training=training)
        o = add([x, y])
        o = self.final_activation(o)
        return o
    
    
class DilatedBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 **kwargs):
        super(DilatedBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.block0 = ConvolutionBlock(self.filters, 3, 1, "SAME", 3, "relu")
        self.block1 = ConvolutionBlock(self.filters, 3, 1, "SAME", 6, "relu")
        self.block2 = ConvolutionBlock(self.filters, 3, 1, "SAME", 9, "relu")
        self.block3 = ConvolutionBlock(self.filters, 1, 1, "SAME", 1, "relu")
        
    def call(self, inputs, training=False):
        x1 = self.block0(inputs, training=training)
        x2 = self.block1(inputs, training=training)
        x3 = self.block2(inputs, training=training)
        x  = concatenate([x1, x2, x3])
        x  = self.block3(x)
        return x
    

class DecodeBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 **kwargs):
        super(DecodeBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.upsample = UpSampling2D((2, 2), interpolation='bilinear')
        self.residual = ResidualBlock(self.filters)
        
    def call(self, inputs, training=False):
        x1, x2 = inputs
        x = self.upsample(x1, training=training)
        x = concatenate([x, x2])
        x = self.residual(x)
        return x
    
    
class TestNet(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 **kwargs):
        super(TestNet, self).__init__(**kwargs)
        self.backbone = backbone

    def build(self, input_shape):
        brigde1 = DilatedBlock(1024)
        
        decode1 = 
    
    def call(self, inputs, training=False):
        S1, S2, S3, S4, S5 = self.backbone(inputs, training=training)
        return x