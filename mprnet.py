from layers import *
import tensorflow as tf


class MPRNet(tf.keras.models.Model):
    def __init__(self,
                 c,
                 ors_scale_c,
                 input_h,
                 input_w,
                 loss_lambda
                 ):
        super(MPRNet, self).__init__()
        self.c = c
        self.ors_scale_c = ors_scale_c
        self.input_h = input_h
        self.input_w = input_w
        self.loss_lambda = loss_lambda

        self.ORS = ORSNet(self.c, self.ors_scale_c)



    @tf.function
    def compute_charbonnier_loss(self, x_s, y):
        return tf.reduce_mean(tf.sqrt((x_s - y)**2 + self.epsilon**2))

    def get_edgeloss_components(self):
        filters = tf.constant([[.05, .25, .4, .25, .05]])
        filters = tf.matmul(tf.transpose(filters), filters)
        filters = tf.expand_dims(filters, axis=2)
        self.filters = tf.stack([filters for _ in range(3)], axis=2)
        strides = [1. if i % 2 == 0 else 0. for i in range(self.input_h)]
        strides = tf.stack([strides if i % 2 == 0 else [0. for _ in range(self.input_h)] for i in range(self.input_w)])
        self.strides = tf.expand_dims(tf.expand_dims(strides, axis=0), axis=3)
        self.gauss_padding = ReplicatePadding2D(2)

    @tf.function
    def conv_gauss(self, img):
        return tf.nn.depthwise_conv2d(self.gauss_padding(ReplicatePadding2D),
                                      self.filters,
                                      strides=[1,1,1,1],
                                      padding='VALID'
                                      )

    @tf.function
    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        filtered = self.conv_gauss(current * self.strides * 4.)
        return current - filtered

    @tf.funciton
    def compute_edge_loss(self, x_s, y):
        return self.loss_lambda * self.compute_charbonnier_loss(self.laplacian_kernel(x_s), self.laplacian_kernel(y))

