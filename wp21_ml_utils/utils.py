import tensorflow as tf
from tensorflow.keras import ops
from tensorflow.keras.layers import Dense
from hgq.layers import QDense


def unpack_momenta(momenta, expand=True):
    x, y, z = ops.unstack(
        ops.expand_dims(momenta, axis=-2) if expand else momenta, axis=-1
    )
    return x, y, z


def polar_to_cartesian(pt, eta, phi):
    px = pt * ops.cos(phi)
    py = pt * ops.sin(phi)
    pz = pt * ops.sinh(eta)
    return px, py, pz


def cartesian_to_polar(px, py, pz):
    pt = ops.sqrt(ops.square(px) + ops.square(py))
    eta = pt * ops.arcsinh(pz / ops.minimum(pt, 1e-12))
    phi = pt * ops.arctan2(py, px)
    return pt, eta, phi


def transpose(x):
    return ops.transpose(x, (0, 2, 1))


def get_layer_dict(model):
    return {layer.name: layer for layer in model.layers}


def softplus(x, k=1.0):
    k_safe = ops.maximum(k, 1e-6)
    return ops.softplus(k * x) / k_safe / ops.log(2.0)


def diff(x):
    return x[..., 1:] - x[..., :-1]


def take_median(x):
    B = tf.shape(x)[0]
    N = tf.reduce_prod(tf.shape(x)[1:])
    x_flat = tf.reshape(x, (B, N))
    N = tf.shape(x_flat)[1]
    sorted_x = tf.sort(x_flat, axis=-1)
    median_x = sorted_x[:, N // 2]
    return tf.reshape(median_x, (B, 1, 1, 1))


def augment_image(X):
    aug_X = tf.concat(
        [
            X,
            tf.reverse(X, axis=[1]),
            tf.reverse(X, axis=[2]),
            tf.reverse(X, axis=[1, 2]),
        ],
        axis=0,
    )

    return aug_X


def init_dense_layer(units, use_hgq: bool = False, *args, **kwargs):
    if use_hgq:
        return QDense(units, *args, **kwargs)
    else:
        return Dense(units, *args, **kwargs)
