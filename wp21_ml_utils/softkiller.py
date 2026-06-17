import tensorflow as tf
from tensorflow.keras.layers import Layer, MaxPooling2D, AveragePooling2D
import numpy as np
from tensorflow.types.experimental import TensorLike


from wp21_ml_utils.utils import take_median


class TowerSoftKiller(Layer):
    """
    SoftKiller pileup-suppression layer for calorimeter tower images.

    Implements the SoftKiller algorithm by dividing the detector image into
    patches, computing the maximum tower transverse energy in each patch, and
    determining the median of these local maxima. Towers with energy below
    the median threshold are removed.

    This procedure suppresses diffuse low-energy background activity while
    preserving localised high-energy deposits.

    Input:
        Tensor of shape (B, E, P, C), typically representing calorimeter
        tower transverse energies.

    Output:
        Tensor of identical shape with low-energy towers set to zero.

    Parameters
    ----------
    patch_size : tuple[int, int], default=(5, 8)
        Size of the η–φ patches used to determine the SoftKiller threshold.
    """

    def __init__(self, patch_size=(5, 8), **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def build(self, input_shape):
        self.max_pooling_layer = MaxPooling2D(
            pool_size=self.patch_size, padding="valid"
        )
        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        local_max = self.max_pooling_layer(image)
        median_max = take_median(local_max)
        sk_towers = tf.where(image > median_max, image, 0)
        return sk_towers

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
        }


class TowerSoftKillerWithAreaCorrection(Layer):
    """
    Area-corrected SoftKiller pileup-suppression layer.

    Performs an initial event-by-event background-density subtraction before
    applying the standard SoftKiller procedure.

    The background density ρ is estimated from the median transverse energy
    density of detector patches:

        ρ = median(E_T^patch / A_patch)

    and subtracted from each tower according to its geometric area. The
    resulting image is then processed using the SoftKiller algorithm to remove
    residual low-energy background activity.

    This approach combines area-based pileup subtraction with SoftKiller
    thresholding and is particularly useful in high-occupancy environments.

    Input:
        Tensor of shape (B, E, P, C), typically representing calorimeter
        tower transverse energies.

    Output:
        Tensor of identical shape with background-subtracted and SoftKiller-
        filtered tower energies.

    Parameters
    ----------
    patch_size : tuple[int, int], default=(5, 8)
        Size of the η–φ patches used for both background-density estimation
        and SoftKiller threshold determination.

    pixel_area : float, default=0.1 * π / 32
        Geometric area associated with a single calorimeter tower in η–φ
        space.
    """

    def __init__(self, patch_size=(5, 8), pixel_area=0.1 * np.pi / 32, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pixel_area = pixel_area

    def build(self, input_shape):
        self.pixels_per_patch = tf.cast(
            self.patch_size[0] * self.patch_size[1], dtype=tf.float32
        )
        self.max_pooling_layer = MaxPooling2D(
            pool_size=self.patch_size, padding="valid"
        )
        self.avg_pooling_layer = AveragePooling2D(pool_size=self.patch_size)

        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        # rho subtraction
        sum_patch_Et = self.pixels_per_patch * self.avg_pooling_layer(image)
        rho = take_median(sum_patch_Et / (self.pixel_area * self.pixels_per_patch))
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
