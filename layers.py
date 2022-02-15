import tensorflow as tf
import tensorflow_addons as tfa


class CABlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 r=16
                 ):
        '''
        Channel-Attention Block
        :param c:
        :param r:
        '''
        super(CABlock, self).__init__()
        self.c = c
        self.r = r

        self.f = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.c,
                                   kernel_size=(3, 3),
                                   use_bias=False,
                                   padding='same',
                                   strides=(1, 1)
                                   ),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Conv2D(filters=self.c,
                                   kernel_size=(3, 3),
                                   use_bias=False,
                                   padding='same',
                                   strides=(1, 1)
                                   )
        ])
        self.attention = tf.keras.Sequential([
            tfa.layers.AdaptiveAveragePooling2D(1),
            tf.keras.layers.Conv2D(filters=self.c // r,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False
                                   ),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Conv2D(filters=self.c,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False,
                                   activation='sigmoid'
                                   )
        ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)
        f = tf.multiply(f, self.attention(f))
        return inputs + f


class SAModule(tf.keras.layers.Layer):
    def __init__(self,
                 c
                 ):
        '''
        Supervised-Attention Module
        :param c:
        '''
        super(SAModule, self).__init__()
        self.c = c

        self.to_rs = tf.keras.layers.Conv2D(3,
                                            kernel_size=1,
                                            strides=1,
                                            padding='valid',
                                            use_bias=False
                                            )
        self.to_attention = tf.keras.layers.Conv2D(self.c,
                                                   kernel_size=1,
                                                   strides=1,
                                                   padding='valid',
                                                   use_bias=False
                                                   )
        self.residual = tf.keras.layers.Conv2D(self.c,
                                               kernel_size=1,
                                               strides=1,
                                               padding='valid',
                                               use_bias=False
                                               )

    def call(self, feature, img):
        recon_img = img + self.to_rs(feature)
        attention = self.to_attention(recon_img)
        return feature + tf.multiply(self.residual(feature), attention), recon_img


class ORBlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 num_cabs=8
                 ):
        super(ORBlock, self).__init__()
        self.c = c
        self.num_cabs = num_cabs

        self.forward = tf.keras.Sequential([
            CABlock(self.c)
        ] + [
            tf.keras.layers.Conv2D(self.c,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False
                                   )
        ])

    def call(self, inputs, **kwargs):
        return inputs + self.forward(inputs)


class Upsample(tf.keras.layers.Layer):
    def __init__(self,
                 c
                 ):
        super(Upsample, self).__init__()
        self.c = c

        self.forward = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=2,
                                         interpolation='bilinear'
                                         ),
            tf.keras.layers.Conv2D(filters=c,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False
                                   )
        ])
        
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


