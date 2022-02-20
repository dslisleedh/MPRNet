from layers import *
import tensorflow as tf


class ORSNet(tf.keras.layers.Layer):
    def __init__(self,
                 epsilon = 1e-3
                 ):
        super(ORSNet, self).__init__()
        self.epsilon = epsilon

    @tf.function
    def compute_charbonnier_loss(self, x_s, y):
        return tf.reduce_mean(tf.sqrt((x_s - y)**2 + self.epsilon**2))
