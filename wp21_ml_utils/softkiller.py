import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


from wp21_ml_utils.utils import take_median


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

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
        }


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

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
            "pixel_area": self.pixel_area,
        }
