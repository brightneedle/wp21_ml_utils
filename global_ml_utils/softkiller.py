import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np


from global_ml_utils.utils import take_median

tfb = tfp.bijectors


class SoftKiller(layers.Layer):
    def __init__(self, patch_size=(5, 8), **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.max_pooling_layer = layers.MaxPooling2D(
            pool_size=patch_size, padding="valid"
        )

    def call(self, image):
        local_max = self.max_pooling_layer(image)
        median_max = take_median(local_max)
        sk_towers = tf.where(image > median_max, image, 0)
        return sk_towers


class SoftKillerWithAreaCorrection(layers.Layer):
    def __init__(self, patch_size=(5, 8), pixel_area=0.1 * np.pi / 32, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.max_pooling_layer = layers.MaxPooling2D(
            pool_size=patch_size, padding="valid"
        )
        self.avg_pooling_layer = layers.AveragePooling2D(pool_size=patch_size)
        self.pixels_per_patch = tf.cast(
            self.patch_size[0] * self.patch_size[1], dtype=tf.float32
        )
        self.pixel_area = pixel_area

    def call(self, image):
        # rho subtraction
        sum_patch_Et = self.pixels_per_patch * self.avg_pooling_layer(image)
        rho = take_median(sum_patch_Et / (self.pixel_area * self.towers_per_patch))
        image_rho_sub = tf.maximum(image - rho * self.pixel_area, 0)

        # normal softkiller
        local_max = self.max_pooling_layer(image_rho_sub)
        median_max = take_median(local_max)
        sk_towers = tf.where(image_rho_sub > median_max, image_rho_sub, 0)
        return sk_towers
