import tensorflow as tf
from tensorflow.keras import losses
import numpy as np

from wp21_ml_utils.utils import unpack, polar_to_cartesian, transpose


def masked_pairwise_distances(p_true, p_pred, pt_weight=1):
    p_true = unpack(p_true)
    p_pred = unpack(p_pred)

    true_pt, pred_pt = p_true[0], p_pred[0]

    true_px, true_py, true_pz = polar_to_cartesian(*p_true)
    pred_px, pred_py, pred_pz = polar_to_cartesian(*p_pred)

    true_mask = true_pt > 0
    pred_mask = pred_pt > 0
    mask = true_mask & transpose(pred_mask)

    dist_pt = tf.square(true_pt - transpose(pred_pt))
    dist_rest = (
        tf.square(true_pz - transpose(pred_pz))
        + 2 * true_pt * transpose(pred_pt)
        - 2 * true_px * transpose(pred_px)
        - 2 * true_py * transpose(pred_py)
    )

    dist = pt_weight * dist_pt + (2 - pt_weight) * dist_rest

    masked_dist = tf.where(mask, dist, tf.constant(np.inf))

    return masked_dist, mask


def chamfer_distance(
    y_true,
    y_pred,
    squared=True,
    reduce_mean=True,
    pt_weight=1,
    normalise=False,
    include_pred_to_true=True,
):
    dists, mask = masked_pairwise_distances(y_true, y_pred, pt_weight=pt_weight)

    T = tf.cast(tf.reduce_any(mask, axis=2), dtype=tf.float32)
    P = tf.cast(tf.reduce_any(mask, axis=1), dtype=tf.float32)

    min_true_to_pred = tf.reduce_min(dists, axis=2)
    min_true_to_pred = tf.where(tf.math.is_inf(min_true_to_pred), 0.0, min_true_to_pred)

    if include_pred_to_true:
        min_pred_to_true = tf.reduce_min(dists, axis=1)
        min_pred_to_true = tf.where(
            tf.math.is_inf(min_pred_to_true), 0.0, min_pred_to_true
        )

    if normalise:
        true_px, true_py, true_pz = polar_to_cartesian(*unpack(y_true))
        p_true_mag_sq = (
            tf.square(true_px) + tf.square(true_py) + tf.square(true_pz) + 1e-12
        )
        p_true_mag_sq = tf.squeeze(p_true_mag_sq, axis=-1)
        min_true_to_pred = min_true_to_pred / p_true_mag_sq
        if include_pred_to_true:
            nearest_true_idx = tf.argmin(dists, axis=1)
            p_true_mag_sq_nearest = tf.gather(
                p_true_mag_sq, nearest_true_idx, batch_dims=1
            )
            min_pred_to_true = min_pred_to_true / p_true_mag_sq_nearest

    if not squared:
        min_true_to_pred = tf.sqrt(tf.maximum(min_true_to_pred, 1e-12))
        if include_pred_to_true:
            min_pred_to_true = tf.sqrt(tf.maximum(min_pred_to_true, 1e-12))

    loss_per_example = tf.reduce_sum(min_true_to_pred, axis=1) / (
        tf.reduce_sum(T, axis=1) + 1e-6
    )
    if include_pred_to_true:
        loss_per_example += tf.reduce_sum(min_pred_to_true, axis=1) / (
            tf.reduce_sum(P, axis=1) + 1e-6
        )

    return tf.reduce_mean(loss_per_example) if reduce_mean else loss_per_example


class ChamferLoss(losses.Loss):
    def __init__(
        self,
        squared=True,
        pt_weight=1,
        scale_by=1.0,
        normalise=False,
        include_pred_to_true=True,
        name="chamfer_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.squared = squared
        self.pt_weight = pt_weight
        self.scale_by = scale_by
        self.normalise = normalise
        self.include_pred_to_true = include_pred_to_true

    def call(self, y_true, y_pred):
        loss = chamfer_distance(
            y_true,
            y_pred,
            squared=self.squared,
            reduce_mean=True,
            pt_weight=self.pt_weight,
            normalise=self.normalise,
            include_pred_to_true=self.include_pred_to_true,
        )
        return self.scale_by * loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "squared": self.squared,
            "scale_by": self.scale_by,
            "pt_weight": self.pt_weight,
            "normalise": self.normalise,
            "include_pred_to_true": self.include_pred_to_true,
        }
        return {**base_config, **config}


class SparsityLoss(losses.Loss):
    def __init__(
        self,
        fmax: float,
        k: int = 1,
        axis: int = (1, 2, 3),
        T: float = 50,
        sum_over_last_axis: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fmax = fmax
        self.k = k
        self.axis = axis
        self.T = T
        self.sum_over_last_axis = sum_over_last_axis

    def call(self, y_true, y_pred):
        weights = y_true
        x = y_pred

        if self.sum_over_last_axis:
            x = tf.reduce_sum(x, axis=-1, keepdims=True)

        is_active = tf.nn.tanh(self.T * tf.abs(x))
        f = tf.reduce_mean(is_active, axis=self.axis)
        penalty = tf.nn.relu(f - self.fmax)
        penalty_norm = 1 - self.fmax

        weights_ = tf.reshape(weights, tf.shape(penalty))

        if self.k == 1:
            mean_penalty = tf.reduce_mean(weights_ * penalty / penalty_norm)

        else:
            mean_penalty = tf.reduce_mean(weights_ * (penalty / penalty_norm) ** self.k)

        return mean_penalty

    def get_config(self):
        return {
            **super().get_config(),
            "fmax": self.fmax,
            "k": self.k,
            "axis": self.axis,
            "T": self.T,
            "sum_over_last_axis": self.sum_over_last_axis,
        }


class CalibrationLoss(losses.Loss):
    def __init__(
        self, max_dR=0.3, squared=True, normalise=False, name="calib_loss", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.max_dR = max_dR
        self.squared = squared
        self.normalise = normalise

    def call(self, y_true, y_pred):
        pt_true, eta_true, phi_true = unpack(y_true)
        pt_pred, eta_pred, phi_pred = unpack(y_pred)

        dpt = tf.abs(pt_true - transpose(pt_pred))

        if self.normalise:
            dpt = dpt / tf.maximum(pt_true, 1e-6)

        if self.squared:
            dpt = dpt**2

        dphi = phi_true - transpose(phi_pred)
        dphi = tf.atan2(tf.sin(dphi), tf.cos(dphi))
        deta = eta_true - transpose(eta_pred)

        dR2 = deta**2 + dphi**2
        matched = (dR2 < self.max_dR**2) & (pt_true > 0) & (transpose(pt_pred) > 0)
        matched = tf.cast(matched, dtype=tf.float32)

        # remove duplication
        is_bijective = (tf.reduce_sum(matched, axis=1, keepdims=True) == 1) & (
            tf.reduce_sum(matched, axis=2, keepdims=True) == 1
        )
        matched = tf.where(is_bijective, matched, 0.0)

        loss = tf.reduce_sum(matched * dpt, axis=(1, 2))
        norm = tf.reduce_sum(matched, axis=(1, 2))
        norm = tf.maximum(norm, 1e-6)
        loss = loss / norm
        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            **super().get_config(),
            "max_dR": self.max_dR,
            "squared": self.squared,
            "normalise": self.normalise,
        }
