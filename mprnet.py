from layers import *
import tensorflow as tf


class MPRNet(tf.keras.models.Model):
    def __init__(self,
                 c,
                 ors_scale_c
                 ):
        super(MPRNet, self).__init__()
        self.c = c
        self.ors_scale_c = ors_scale_c

        self.ORS = ORSNet(self.c, self.ors_scale_c)

    @tf.function
    def compute_charbonnier_loss(self, x_s, y):
        return tf.reduce_mean(tf.sqrt((x_s - y)**2 + self.epsilon**2))
