import tensorflow as tf
from tensorflow.keras import layers, backend
import numpy as np

from wp21_ml_utils.utils import unpack
from wp21_ml_utils.layers import TowerEtaPhiLayer


class ImageToVectors(layers.Layer):
    def __init__(self, max_vectors: int = None, min_pt: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_vectors = max_vectors
        self.min_pt = min_pt
        self.get_coords = TowerEtaPhiLayer()

    def call(self, image):
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta, phi = self.get_coords(image)

        pt = backend.sum(image, axis=-1, keepdims=True)

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
        }


class VectorsToImage(layers.Layer):
    def __init__(
        self,
        eta_edges: tf.Tensor = tf.linspace(-2.5, 2.5, 51),
        phi_edges: tf.Tensor = tf.linspace(-np.pi, np.pi, 65),
    ):
        super().__init__()
        self.eta_edges = tf.cast(tf.convert_to_tensor(eta_edges), tf.float32)
        self.phi_edges = tf.cast(tf.convert_to_tensor(phi_edges), tf.float32)

    def call(self, x):
        pt, eta, phi = unpack(x, expand=False)

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

        indices = tf.stack(
            [
                tf.reshape(batch_idx, [-1]),
                tf.reshape(eta_bin, [-1]),
                tf.reshape(phi_bin, [-1]),
            ],
            axis=1,
        )

        updates = tf.reshape(pt, [-1])

        towers = tf.zeros((B, n_eta, n_phi), dtype=tf.float32)
        towers = tf.tensor_scatter_nd_add(towers, indices, updates)

        towers = tf.expand_dims(towers, axis=-1)

        return towers

    def get_config(self):
        return {
            **super().get_config(),
            "eta_edges": self.eta_edges,
            "phi_edges": self.phi_edges,
        }
