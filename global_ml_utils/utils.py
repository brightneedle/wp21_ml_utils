import tensorflow as tf


def unpack(momenta):
    return tf.unstack(tf.expand_dims(momenta, axis=-1), axis=-2)


def polar_to_cartesian(pt, eta, phi):
    px = pt * tf.math.cos(phi)
    py = pt * tf.math.sin(phi)
    pz = pt * tf.math.sinh(eta)
    return px, py, pz


def transpose(x):
    return tf.transpose(x, (0, 2, 1))


def get_layer_dict(model):
    return {layer.name: layer for layer in model.layers}


def softplus(x, k=1.0):
    k_safe = tf.maximum(k, 1e-6)
    return tf.nn.softplus(k * x) / k_safe / tf.math.log(2.0)


def smooth_piecewise_linear(x, left_slope, right_slope, k=1):
    return left_slope * softplus(x, k) - right_slope * softplus(-x, k)


def smooth_piecewise_linear_deriv(x, left_slope, right_slope, k=1):
    k_safe = tf.maximum(k, 1e-6)
    return (
        left_slope * tf.nn.sigmoid(k_safe * x)
        + right_slope * tf.nn.sigmoid(-k_safe * x)
    ) / tf.math.log(2.0)


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
