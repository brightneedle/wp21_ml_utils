import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

from wp21_ml_utils.layers import SlidingConeSum, LocalMaxMask
from wp21_ml_utils.converters import ImageToVectors


@register_keras_serializable("wp21_ml_utils")
class ConeJet(Layer):
    """
    Differentiable cone-based jet reconstruction layer from calorimeter images.

    This layer implements a simplified cone jet-finding procedure directly on
    η–φ calorimeter images. It combines local energy aggregation, seed finding
    via local maxima, and vectorisation into a fixed-size jet representation.

    The algorithm proceeds in three stages:

    1. Cone energy accumulation:
       A sliding cone kernel computes the summed transverse energy within a
       geometrical cone (circular or custom shape) around each pixel.

    2. Jet seed selection:
       Local maxima are identified using a neighbourhood-based peak filter.
       Only pixels that are local maxima are considered as valid jet seeds.

    3. Seeded cone masking:
       Cone sums are retained only at seed locations; all other responses are
       suppressed.

    4. Vectorisation:
       The resulting sparse jet image is converted into a fixed-size set of
       jet vectors sorted/truncated by transverse momentum.

    Input:
        Tensor of shape (B, H, W, C), typically representing calorimeter
        tower energies in η–φ space.

    Output:
        Tensor of shape (B, max_jets, F), where each entry corresponds to a
        reconstructed jet feature vector (e.g. pT and possibly additional
        attributes depending on ImageToVectors).

    Parameters
    ----------
    kernel_size : int, default=9
        Size of the sliding cone window used for local energy aggregation.

    min_pt : float, default=0
        Minimum transverse momentum threshold for jet acceptance in the final
        vectorisation step.

    max_jets : int, default=20
        Maximum number of jets returned per event. Jets are typically ordered
        by descending transverse momentum.

    shape : str, default="circle"
        Geometrical shape of the cone kernel. Common options include "circle"
        and potentially other detector-motivated shapes supported by
        SlidingConeSum.

    radius : float, optional
        Physical radius of the cone in η–φ space. If None, defaults to a value
        derived from kernel_size.
    """

    def __init__(
        self,
        kernel_size: int = 9,
        min_pt: float = 0,
        max_jets: int = 20,
        shape: str = "circle",
        radius: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.min_pt = min_pt
        self.max_jets = max_jets
        self.shape = shape
        self.radius = radius

    def build(self, input_shape):
        self.cone_sum = SlidingConeSum(
            kernel_size=self.kernel_size,
            shape=self.shape,
            radius=self.radius,
            name="cone_sum",
        )

        self.local_max = LocalMaxMask(
            kernel_size=self.kernel_size,
            shape=self.shape,
            name="local_max",
        )

        self.image_to_vectors = ImageToVectors(
            max_vectors=self.max_jets,
            min_pt=self.min_pt,
        )

        super().build(input_shape)

    def call(self, image):
        sum_image = tf.reduce_sum(image, axis=-1, keepdims=True)
        cone_sums = self.cone_sum(sum_image)

        local_max_seed_mask = self.local_max(sum_image)

        masked_cone_sums = tf.where(
            local_max_seed_mask,
            cone_sums,
            0,
        )

        return self.image_to_vectors(masked_cone_sums)
