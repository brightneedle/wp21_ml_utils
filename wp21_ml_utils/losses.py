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
    normalise_by_truth: bool = False,
    normalise_by_multiplicity: bool = False,
    include_pred_to_true: bool = True,
) -> tf.Tensor:
    """
    Computes a masked Chamfer distance between particle collections.

    Each particle in the ground-truth collection is matched to its nearest
    predicted particle, and optionally each predicted particle is matched to
    its nearest ground-truth particle. Invalid or padded particles are
    excluded from the matching.

    The distance between particles is defined by
    :func:`masked_pairwise_distances` and can include a relative weighting
    between the pT and momentum-space components. Distances may be computed
    either as squared distances or Euclidean distances.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth particle collection. The particle dimension is expected
        to be the penultimate dimension, with particle features in the final
        dimension.

    y_pred : tf.Tensor
        Predicted particle collection with the same feature representation
        as ``y_true``.

    squared : bool, default=True
        If True, use squared distances. If False, take the square root of the
        squared distances to obtain Euclidean distances.

    reduce_mean : bool, default=True
        If True, return the mean Chamfer distance over the batch. If False,
        return one loss value per event.

    pt_weight : float, default=1.0
        Relative weight applied to the pT component of the particle distance.
        The precise distance definition is determined by
        :func:`masked_pairwise_distances`.

    normalise_by_truth : bool, default=False
        If True, divide each matched distance by the squared momentum
        magnitude of the corresponding ground-truth particle. For the
        predicted-to-true term, the normalisation uses the momentum magnitude
        of the nearest ground-truth particle.

        This provides a relative rather than absolute momentum-space distance.
        In particular, when ``squared=True`` the squared distance is divided
        by the squared momentum magnitude.

    normalise_by_multiplicity : bool, default=False
        If True, normalise each matching direction by the number of valid
        particles in that collection. This makes the contribution approximately
        independent of the number of particles in an event.

    include_pred_to_true : bool, default=True
        If True, include both ground-truth-to-predicted and
        predicted-to-ground-truth matching terms, giving a symmetric Chamfer
        distance. If False, only the ground-truth-to-predicted term is
        included, making the distance asymmetric.

    Returns
    -------
    tf.Tensor
        If ``reduce_mean=True``, a scalar containing the mean Chamfer distance
        over the batch. Otherwise, a tensor containing one loss value per
        event.
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

    if normalise_by_truth:
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

    D_true_to_pred = tf.reduce_sum(min_true_to_pred, axis=1)
    if normalise_by_multiplicity:
        D_true_to_pred /= tf.maximum(tf.reduce_sum(T, axis=1), 1e-6)

    if include_pred_to_true:
        D_pred_to_true = tf.reduce_sum(min_pred_to_true, axis=1)
        if normalise_by_multiplicity:
            D_pred_to_true /= tf.maximum(tf.reduce_sum(P, axis=1), 1e-6)

        loss = D_true_to_pred + D_pred_to_true

    else:
        loss = D_true_to_pred

    return tf.reduce_mean(loss) if reduce_mean else loss


@register_keras_serializable("wp21_ml_utils")
class ChamferLoss(Loss):
    """
    Keras loss implementing a masked Chamfer distance between particle
    collections.

    The loss compares predicted and target particle collections by matching
    each valid particle to its nearest neighbour in the opposite collection.
    It supports both symmetric and asymmetric Chamfer distances, squared or
    Euclidean distances, momentum-dependent normalisation, and normalisation
    by particle multiplicity.

    This loss is suitable for collections with padded or otherwise invalid
    particle entries, which are ignored during the matching.

    Parameters
    ----------
    squared : bool, default=True
        If True, use squared distances. If False, use Euclidean distances.

    pt_weight : float, default=1.0
        Relative weight applied to the pT component of the particle distance.
        The complete distance definition is provided by
        :func:`masked_pairwise_distances`.

    normalise_by_truth : bool, default=False
        If True, normalise each matched distance by the squared momentum
        magnitude of the corresponding ground-truth particle. For the
        predicted-to-true term, the momentum magnitude of the nearest
        ground-truth particle is used.

    normalise_by_multiplicity : bool, default=False
        If True, divide each matching direction by the number of valid
        particles in the corresponding collection.

    include_pred_to_true : bool, default=True
        If True, include both ground-truth-to-predicted and
        predicted-to-ground-truth matching terms. If False, only the
        ground-truth-to-predicted term is included.

    **kwargs
        Additional keyword arguments passed to
        :class:`keras.losses.Loss`.

    Notes
    -----
    The loss returns a scalar batch-averaged value, since ``call`` evaluates
    :func:`chamfer_distance` with ``reduce_mean=True``.
    """

    def __init__(
        self,
        squared: bool = True,
        pt_weight: float = 1.0,
        normalise_by_truth: bool = False,
        normalise_by_multiplicity: bool = False,
        include_pred_to_true: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squared = squared
        self.pt_weight = float(pt_weight)
        self.normalise_by_truth = normalise_by_truth
        self.normalise_by_multiplicity = normalise_by_multiplicity
        self.include_pred_to_true = include_pred_to_true

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = chamfer_distance(
            y_true,
            y_pred,
            squared=self.squared,
            reduce_mean=True,
            pt_weight=self.pt_weight,
            normalise_by_truth=self.normalise_by_truth,
            normalise_by_multiplicity=self.normalise_by_multiplicity,
            include_pred_to_true=self.include_pred_to_true,
        )
        return loss

    def get_config(self):
        base_config = super().get_config()
        config = {
            "squared": self.squared,
            "pt_weight": self.pt_weight,
            "normalise_by_truth": self.normalise_by_truth,
            "normalise_by_multiplicity": self.normalise_by_multiplicity,
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
