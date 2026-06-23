from typing import Union

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable
import numpy as np

from wp21_ml_utils.utils import unpack_momenta, polar_to_cartesian, transpose


def masked_pairwise_distances(
    p_true: tf.Tensor, p_pred: tf.Tensor, pt_weight: float = 1.0
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Computes masked pairwise distances between sets of particles.

    Particles are represented in polar coordinates (pT, η, φ) and converted
    to Cartesian momentum components before distance evaluation. Distances are
    only computed between valid particles with positive transverse momentum.

    The distance metric combines differences in pT and momentum-vector
    separation:

        d = pt_weight · d_pt + (2 - pt_weight) · d_momentum

    Invalid particle pairs are assigned infinite distance.

    Parameters
    ----------
    p_true : tf.Tensor
        Ground-truth particles of shape (B, N_true, F).

    p_pred : tf.Tensor
        Predicted particles of shape (B, N_pred, F).

    pt_weight : float, default=1.0
        Relative weighting between transverse-momentum and momentum-space
        distance contributions.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        Pairwise distance matrix and validity mask, both of shape
        (B, N_true, N_pred).
    """
    p_true = unpack_momenta(p_true[..., :3])
    p_pred = unpack_momenta(p_pred[..., :3])

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
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    squared: bool = True,
    reduce_mean: bool = True,
    pt_weight: float = 1.0,
    normalise: bool = False,
    include_pred_to_true: bool = True,
) -> tf.Tensor:
    """
    Computes a masked Chamfer distance between particle collections.

    For each particle, the nearest neighbour in the opposite collection is
    identified and the corresponding distance accumulated. Optionally includes
    both true→predicted and predicted→true matching directions.

    Supports momentum-magnitude normalisation and either squared or Euclidean
    distances.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth particle collection.

    y_pred : tf.Tensor
        Predicted particle collection.

    squared : bool, default=True
        If True, uses squared distances.

    reduce_mean : bool, default=True
        If True, returns the batch mean. Otherwise returns one loss value
        per event.

    pt_weight : float, default=1.0
        Relative weighting of pT and momentum-space distance terms.

    normalise : bool, default=False
        Normalises distances by the momentum magnitude of the matched
        ground-truth particle.

    include_pred_to_true : bool, default=True
        Includes the reverse matching direction in the Chamfer distance.

    Returns
    -------
    tf.Tensor
        Scalar loss or per-example loss values depending on
        ``reduce_mean``.
    """
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
        true_px, true_py, true_pz = polar_to_cartesian(*unpack_momenta(y_true[..., :3]))
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


@register_keras_serializable("wp21_ml_utils")
class ChamferLoss(Loss):
    """
    Keras loss wrapper for particle-level Chamfer distance.

    Measures the similarity between predicted and target particle
    collections by matching each particle to its nearest neighbour in
    momentum space. Supports asymmetric and symmetric variants of the
    Chamfer distance as well as optional momentum normalisation.

    Parameters
    ----------
    squared : bool, default=True
        Uses squared distances when True.

    pt_weight : float, default=1.0
        Relative weighting of pT and momentum-space distance terms.

    normalise : bool, default=False
        Normalises distances by the momentum magnitude of the matched
        target particle.

    include_pred_to_true : bool, default=True
        Includes the predicted→true matching contribution.
    """

    def __init__(
        self,
        squared: bool = True,
        pt_weight: float = 1.0,
        normalise: bool = False,
        include_pred_to_true: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squared = squared
        self.pt_weight = float(pt_weight)
        self.normalise = normalise
        self.include_pred_to_true = include_pred_to_true

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = chamfer_distance(
            y_true,
            y_pred,
            squared=self.squared,
            reduce_mean=True,
            pt_weight=self.pt_weight,
            normalise=self.normalise,
            include_pred_to_true=self.include_pred_to_true,
        )
        return loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "squared": self.squared,
            "pt_weight": self.pt_weight,
            "normalise": self.normalise,
            "include_pred_to_true": self.include_pred_to_true,
        }
        return {**base_config, **config}


@register_keras_serializable("wp21_ml_utils")
class SparsityLoss(Loss):
    """
    Sparsity regularisation loss.

    Penalises activations when the fraction of active elements exceeds a
    specified maximum occupancy. Activity is estimated using a smooth
    tanh-based approximation to a binary activation indicator.

    This loss is useful for encouraging sparse detector images, latent
    representations, or object maps.

    Parameters
    ----------
    fmax : float
        Maximum allowed active fraction.

    k : int, default=1
        Exponent applied to the sparsity penalty.

    axis : int or tuple[int], default=(1, 2, 3)
        Axes over which occupancy is computed.

    T : float, default=50
        Temperature controlling the sharpness of the activity indicator.

    sum_over_last_axis : bool, default=False
        Sums channels before computing sparsity.
    """

    def __init__(
        self,
        fmax: float,
        k: int = 1,
        axis: Union[int, tuple[int]] = (1, 2, 3),
        T: float = 50,
        sum_over_last_axis: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fmax = float(fmax)
        self.k = int(k)
        self.axis = axis
        self.T = float(T)
        self.sum_over_last_axis = sum_over_last_axis

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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


@register_keras_serializable("wp21_ml_utils")
class CalibrationLoss(Loss):
    """
    Momentum calibration loss for matched particle pairs.

    Matches predicted and target particles within a configurable ΔR cone and
    penalises differences in transverse momentum. Only one-to-one matches are
    retained; ambiguous matches are discarded.

    This loss is intended for evaluating momentum-scale calibration after
    object localisation has already been learned.

    Parameters
    ----------
    max_dR : float, default=0.3
        Maximum matching distance in η–φ space.

    squared : bool, default=True
        Uses squared pT residuals when True.

    normalise : bool, default=False
        Normalises pT residuals by the target pT.
    """

    def __init__(
        self,
        max_dR: float = 0.3,
        squared: bool = True,
        normalise: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_dR = float(max_dR)
        self.squared = squared
        self.normalise = normalise

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        pt_true, eta_true, phi_true = unpack_momenta(y_true[..., :3])
        pt_pred, eta_pred, phi_pred = unpack_momenta(y_pred[..., :3])

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


@register_keras_serializable("wp21_ml_utils")
class PinballLoss(Loss):
    """
    Quantile regression loss.

    Implements the asymmetric pinball loss used for estimating conditional
    quantiles. Underestimation and overestimation are penalised differently
    according to the target quantile.

    A model trained with this loss converges toward predicting the specified
    quantile of the target distribution.

    Parameters
    ----------
    target_quantile : float
        Target quantile in the open interval (0, 1).

        Examples:
            0.50 -> median
            0.16 -> lower one-sigma equivalent
            0.84 -> upper one-sigma equivalent
    """

    def __init__(self, target_quantile: float, **kwargs):
        super().__init__(**kwargs)
        self.target_quantile = float(target_quantile)
        if self.target_quantile <= 0 or self.target_quantile >= 1:
            raise ValueError(
                f"target quantile must be 0 < q < 1 - got {target_quantile}"
            )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_true - y_pred
        loss = tf.maximum(self.target_quantile * err, (self.target_quantile - 1) * err)
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        return {**super().get_config(), "target_quantile": self.target_quantile}
