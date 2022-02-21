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
        attention = tf.nn.sigmoid(self.to_attention(recon_img))
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
            CABlock(self.c) for _ in range(self.num_cabs)
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
                 c,
                 rate=2
                 ):
        super(Upsample, self).__init__()
        self.c = c
        self.rate = rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=self.rate,
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


class Downsample(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 rate=2
                 ):
        super(Downsample, self).__init__()
        self.c = c
        self.rate = rate

        self.Conv = tf.keras.layers.Conv2D(filters=self.c,
                                           kernel_size=1,
                                           strides=1,
                                           padding='valid',
                                           use_bias=False
                                           )

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        inputs = tf.image.resize(inputs,
                                 size=[h//self.rate, w//self.rate]
                                 )
        return self.Conv(inputs)


class ReplicatePadding2D(tf.keras.layers.Layer):
    def __init__(self, n_pad):
        super(ReplicatePadding2D, self).__init__()
        self.n_pad = n_pad

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        top = tf.concat([inputs[:, :1, :, :] for _ in range(self.n_pad)],
                        axis=1
                        )
        bottom = tf.concat([inputs[:, h-1:, :, :] for _ in range(self.n_pad)],
                           axis=1
                           )
        inputs = tf.concat([top, inputs, bottom],
                           axis=1
                           )
        left = tf.concat([inputs[:, :, :1, :] for _ in range(self.n_pad)],
                         axis=2
                         )
        right = tf.concat([inputs[:, :, w-1:, :] for _ in range(self.n_pad)],
                          axis=2
                          )
        inputs = tf.concat([left, inputs, right],
                           axis=2
                           )
        return inputs


################## Modules #################


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return

class ORSNet(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 scale_c,
                 epsilon=1e-3
                 ):
        super(ORSNet, self).__init__()
        self.c = c
        self.scale_c = scale_c
        self.epsilon = epsilon

        self.Orb1 = ORBlock(self.c+self.scale_c)
        self.Orb2 = ORBlock(self.c+self.scale_c)
        self.Orb3 = ORBlock(self.c+self.scale_c)

        self.UpEnc1 = Upsample(self.scale_c)
        self.UpDec1 = Upsample(self.scale_c)
        self.UpEnc2 = tf.keras.Sequential([
            Upsample(self.scale_c+self.scale_c),
            Upsample(self.scale_c)
        ])
        self.UpDec2 = tf.keras.Sequential([
            Upsample(self.scale_c+self.scale_c),
            Upsample(self.scale_c)
        ])

        self.ConvEnc1 = self.construct_layerconv()
        self.ConvDec1 = self.construct_layerconv()
        self.ConvEnc2 = self.construct_layerconv()
        self.ConvDec2 = self.construct_layerconv()
        self.ConvEnc3 = self.construct_layerconv()
        self.ConvDec3 = self.construct_layerconv()

    def construct_layerconv(self):
        return tf.keras.layers.Conv2D(self.c + self.scale_c,
                                      kernel_size=1,
                                      padding='same',
                                      activation='linear',
                                      use_bias=False
                                      )

    def call(self, inputs, encoder_outputs, decoder_outputs):
        inputs = self.Orb1(inputs)
        inputs = inputs + self.ConvEnc1(encoder_outputs[0]) + self.ConvDec1(decoder_outputs[0])

        inputs = self.Orb2(inputs)
        inputs = inputs + self.ConvEnc2(self.UpEnc1(encoder_outputs[1])) + self.ConvDec2(self.UpDec1(decoder_outputs[1]))

        inputs = self.Orb3(inputs)
        inputs = inputs + self.ConvEnc3(self.UpEnc2(encoder_outputs[2])) + self.ConvDec3(self.UpDec2(decoder_outputs[2]))
        return inputs