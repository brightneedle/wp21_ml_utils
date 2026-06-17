from typing import Union

import tensorflow as tf
from tensorflow.keras import ops
from tensorflow.keras.layers import Dense
from hgq.layers import QDense


def unpack_momenta(momenta: tf.Tensor, keepdims: bool = True) -> tuple[tf.Tensor]:
    """
    Splits momentum tensors into individual components.

    Assumes the final dimension contains momentum coordinates ordered as

        (pT, η, φ)

    and returns each component as a separate tensor.

    Parameters
    ----------
    momenta : tf.Tensor
        Momentum tensor with the final dimension corresponding to momentum
        components.

    keepdims : bool, default=True
        Preserves a singleton feature dimension in the returned tensors.

    Returns
    -------
    tuple[tf.Tensor, ...]
        Individual momentum component tensors.
    """
    return ops.unstack(
        ops.expand_dims(momenta, axis=-2) if keepdims else momenta, axis=-1
    )


def polar_to_cartesian(
    pt: tf.Tensor, eta: tf.Tensor, phi: tf.Tensor
) -> tuple[tf.Tensor]:
    """
    Converts momentum coordinates from (pT, η, φ) to (px, py, pz).

    Parameters
    ----------
    pt : tf.Tensor
        Transverse momentum.

    eta : tf.Tensor
        Pseudorapidity.

    phi : tf.Tensor
        Azimuthal angle.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Cartesian momentum components (px, py, pz).
    """
    px = pt * ops.cos(phi)
    py = pt * ops.sin(phi)
    pz = pt * ops.sinh(eta)
    return px, py, pz


def cartesian_to_polar(px: tf.Tensor, py: tf.Tensor, pz: tf.Tensor) -> tuple[tf.Tensor]:
    """
    Converts momentum coordinates from (px, py, pz) to (pT, η, φ).

    Parameters
    ----------
    px, py, pz : tf.Tensor
        Cartesian momentum components.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Polar momentum coordinates (pT, η, φ).
    """
    pt = ops.sqrt(ops.square(px) + ops.square(py))
    eta = pt * ops.arcsinh(pz / ops.minimum(pt, 1e-12))
    phi = pt * ops.arctan2(py, px)
    return pt, eta, phi


def transpose(x: tf.Tensor) -> tf.Tensor:
    """
    Transposes the final two non-batch dimensions.

    Converts tensors of shape

        (B, N, M)

    into

        (B, M, N).

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        Transposed tensor.
    """
    return ops.transpose(x, (0, 2, 1))


def get_layer_dict(model: tf.keras.Model) -> dict:
    """
    Creates a lookup dictionary of model layers.

    Parameters
    ----------
    model : tf.keras.Model
        Model whose layers should be indexed.

    Returns
    -------
    dict
        Mapping from layer names to layer objects.
    """
    return {layer.name: layer for layer in model.layers}


def scaled_softplus(x: tf.Tensor, k: float = 1.0) -> tf.Tensor:
    """
    Scaled SoftPlus activation.

    Computes

        softplus(kx) / (k log 2)

    which approaches ReLU for large values of ``k`` while remaining smooth
    and strictly positive.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    k : float, default=1.0
        Softness parameter controlling the transition sharpness.

    Returns
    -------
    tf.Tensor
        Transformed tensor.
    """
    k_safe = ops.maximum(k, 1e-6)
    return ops.softplus(k * x) / k_safe / ops.log(2.0)


def diff(x: tf.Tensor) -> tf.Tensor:
    """
    Computes first-order finite differences along the final axis.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        Tensor containing adjacent differences.
    """
    return x[..., 1:] - x[..., :-1]


def take_median(x: tf.Tensor) -> tf.Tensor:
    """
    Computes the median value of each batch element.

    All non-batch dimensions are flattened before evaluating the median.
    The result is reshaped to be broadcast-compatible with image-like
    tensors.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        Median values with shape (B, 1, 1, 1).
    """
    B = tf.shape(x)[0]
    N = tf.reduce_prod(tf.shape(x)[1:])
    x_flat = tf.reshape(x, (B, N))
    N = tf.shape(x_flat)[1]
    sorted_x = tf.sort(x_flat, axis=-1)
    median_x = sorted_x[:, N // 2]
    return tf.reshape(median_x, (B, 1, 1, 1))


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Generates reflection-based image augmentations.

    For each input image, produces:

    * original image
    * η reflection
    * φ reflection
    * η–φ reflection

    and concatenates them along the batch dimension.

    Parameters
    ----------
    image : tf.Tensor
        Input image tensor.

    Returns
    -------
    tf.Tensor
        Augmented image batch with four times the original batch size.
    """
    aug_image = tf.concat(
        [
            image,
            tf.reverse(image, axis=[1]),
            tf.reverse(image, axis=[2]),
            tf.reverse(image, axis=[1, 2]),
        ],
        axis=0,
    )

    return aug_image


def init_dense_layer(
    units: int, use_hgq: bool = False, *args, **kwargs
) -> Union[Dense, QDense]:
    """
    Factory function for dense-layer creation.

    Returns either a standard Keras dense layer or an HGQ quantised dense
    layer depending on the value of ``use_hgq``.

    Parameters
    ----------
    units : int
        Number of output units.

    use_hgq : bool, default=False
        If True, returns an HGQ ``QDense`` layer. Otherwise returns a
        standard Keras ``Dense`` layer.

    Returns
    -------
    Dense or QDense
        Constructed dense layer instance.
    """
    if use_hgq:
        return QDense(units, *args, **kwargs)
    else:
        return Dense(units, *args, **kwargs)
