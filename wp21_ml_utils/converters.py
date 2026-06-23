import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.types.experimental import TensorLike
import numpy as np

from wp21_ml_utils.utils import unpack_momenta
from wp21_ml_utils.layers import TowerEtaPhiLayer


@register_keras_serializable("wp21_ml_utils")
class ImageToVectors(Layer):
    """
    Converts a calorimeter-style image tensor into a ranked list of physics-like vectors.

    This layer interprets a structured input image (e.g., eta–phi grid with feature channels)
    and extracts the most significant "seed" cells based on transverse momentum (pT).
    Each selected cell is converted into a vector of the form:

        (pt, eta, phi)

    Key steps:
    - Computes (eta, phi) coordinates for each image cell using a fixed tower geometry.
    - Aggregates per-cell transverse momentum from input channels.
    - Flattens spatial dimensions and selects top-K highest-pT entries.
    - Applies a minimum pT threshold mask.
    - Returns a padded tensor of vectors sorted by descending pT.

    Args:
        max_vectors (int, optional):
            Maximum number of vectors to return per event. If None, all cells are used.
        min_pt (float):
            Minimum transverse momentum threshold below which vectors are zeroed out.
        dphi (float):
            Azimuthal bin size used in coordinate construction.
        deta (float):
            Pseudorapidity bin size used in coordinate construction.
        **kwargs:
            Standard Keras layer keyword arguments.

    Input shape:
        Tensor of shape (B, E, P, C), where:
            B = batch size
            E = eta bins
            P = phi bins
            C = feature channels

    Output shape:
        Tensor of shape (B, K, 3), where:
            K = max_vectors (or E*P if None)
            3 = (pt, eta, phi)
    """

    def __init__(
        self,
        max_vectors: int = None,
        min_pt: float = 0,
        dphi: float = np.pi / 32,
        deta: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_vectors = max_vectors
        self.min_pt = min_pt
        self.dphi = dphi
        self.deta = deta

    def build(self, input_shape):
        self.get_coords = TowerEtaPhiLayer(deta=self.deta, dphi=self.dphi)
        super().build(input_shape)

    def call(self, image: TensorLike) -> tf.Tensor:
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta, phi = self.get_coords(image)

        pt = tf.reduce_sum(image, axis=-1, keepdims=True)

        flat_pt = tf.reshape(pt, [B, E * P])
        flat_eta = tf.reshape(eta, [B, E * P])
        flat_phi = tf.reshape(phi, [B, E * P])

        seed_pt, idxs = tf.math.top_k(
            flat_pt,
            k=E * P if self.max_vectors is None else self.max_vectors,
            sorted=True,
        )

        seed_eta = tf.gather(flat_eta, idxs, axis=1, batch_dims=1)
        seed_phi = tf.gather(flat_phi, idxs, axis=1, batch_dims=1)

        vectors = tf.stack([seed_pt, seed_eta, seed_phi], axis=-1)
        seed_mask = seed_pt[..., None] > self.min_pt

        return tf.where(seed_mask, vectors, 0)

    def get_config(self):
        return {
            **super().get_config(),
            "max_vectors": self.max_vectors,
            "min_pt": self.min_pt,
            "deta": self.deta,
            "dphi": self.dphi,
        }


@register_keras_serializable("wp21_ml_utils")
class VectorsToImage(Layer):
    """
    Converts sparse particle-like vectors into a dense eta–phi (optionally layered) image.

    This layer takes a set of reconstructed physics objects described by (pt, eta, phi)
    coordinates and accumulates them into a fixed 2D grid (or 3D grid with detector layers).

    Each vector contributes its transverse momentum (pT) to the corresponding
    (eta, phi) bin, optionally separated by detector layer.

    Key steps:
    - Unpacks input vectors into (pt, eta, phi [, layer]).
    - Discretizes continuous coordinates into histogram bins using predefined edges.
    - Filters out-of-range values.
    - Scatters pT values into a dense grid using indexed accumulation.
    - Optionally expands output for compatibility with image-like pipelines.

    Args:
        eta_edges (TensorLike):
            Bin edges defining pseudorapidity segmentation.
        phi_edges (TensorLike):
            Bin edges defining azimuthal segmentation.
        n_layers (int, optional):
            Number of detector layers (required if use_layers=True).
        use_layers (bool):
            If True, expects input vectors to include a layer index and produces
            a 4D output tensor.

    Input shape:
        If use_layers=False:
            (B, N, 3) -> (pt, eta, phi)
        If use_layers=True:
            (B, N, 4) -> (pt, eta, phi, layer)

    Output shape:
        If use_layers=False:
            (B, n_eta, n_phi, 1)
        If use_layers=True:
            (B, n_eta, n_phi, n_layers)
    """

    def __init__(
        self,
        eta_edges: TensorLike = np.linspace(-2.5, 2.5, 51),
        phi_edges: TensorLike = np.linspace(-np.pi, np.pi, 65),
        n_layers: int = None,
        return_layers: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eta_edges = tf.cast(tf.convert_to_tensor(eta_edges), tf.float32)
        self.phi_edges = tf.cast(tf.convert_to_tensor(phi_edges), tf.float32)
        self.return_layers = return_layers
        self.n_layers = n_layers

    def call(self, vectors: TensorLike) -> tf.Tensor:
        if self.return_layers:
            pt, eta, phi, layer = unpack_momenta(vectors[..., :4], keepdims=False)
        else:
            pt, eta, phi = unpack_momenta(vectors[..., :3], keepdims=False)
            layer = None

        B = tf.shape(pt)[0]
        N = tf.shape(pt)[1]

        n_eta = tf.shape(self.eta_edges)[0] - 1
        n_phi = tf.shape(self.phi_edges)[0] - 1

        eta_flat = tf.reshape(eta, [-1])
        phi_flat = tf.reshape(phi, [-1])

        eta_bin = tf.searchsorted(self.eta_edges, eta_flat, side="right") - 1
        phi_bin = tf.searchsorted(self.phi_edges, phi_flat, side="right") - 1

        eta_bin = tf.reshape(eta_bin, [B, N])
        phi_bin = tf.reshape(phi_bin, [B, N])

        valid = (eta_bin >= 0) & (eta_bin < n_eta) & (phi_bin >= 0) & (phi_bin < n_phi)

        eta_bin = tf.where(valid, eta_bin, 0)
        phi_bin = tf.where(valid, phi_bin, 0)
        pt = tf.where(valid, pt, 0.0)

        batch_idx = tf.reshape(tf.range(B), (B, 1))
        batch_idx = tf.tile(batch_idx, (1, N))

        if layer is not None:
            layer = tf.cast(layer, tf.int32)

            layer = tf.reshape(layer, [-1])
            layer = tf.reshape(layer, [B, N])

            layer = tf.where(valid, layer, 0)

            indices = tf.stack(
                [
                    tf.reshape(batch_idx, [-1]),
                    tf.reshape(eta_bin, [-1]),
                    tf.reshape(phi_bin, [-1]),
                    tf.reshape(layer, [-1]),
                ],
                axis=1,
            )

            out_shape = (B, n_eta, n_phi, self.n_layers)

        else:
            indices = tf.stack(
                [
                    tf.reshape(batch_idx, [-1]),
                    tf.reshape(eta_bin, [-1]),
                    tf.reshape(phi_bin, [-1]),
                ],
                axis=1,
            )

            out_shape = (B, n_eta, n_phi)

        updates = tf.reshape(pt, [-1])

        towers = tf.zeros(out_shape, dtype=tf.float32)
        towers = tf.tensor_scatter_nd_add(towers, indices, updates)

        if layer is None:
            towers = tf.expand_dims(towers, axis=-1)

        return towers

    def get_config(self):
        return {
            **super().get_config(),
            "eta_edges": self.eta_edges.numpy().tolist(),
            "phi_edges": self.phi_edges.numpy().tolist(),
            "return_layers": self.return_layers,
            "n_layers": self.n_layers,
        }
