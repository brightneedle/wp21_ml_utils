WP2.1 Machine Learning Utilities 🔧
===================================

A compact TensorFlow/Keras utility package for HGQ-based ML studies.

The package provides serialisable Keras layers, callable network blocks,
losses, regularisers, and configuration helpers for building detector-inspired
pipelines that operate on both sparse object vectors and dense eta-phi images.

Features
--------

- Custom Keras layers and reusable callable network blocks for quantisation,
  eta-phi image conversion, pileup suppression, cone-jet reconstruction,
  particle calibration, and common dense/convolutional architectures.
- Support for image-style event inputs, ``B x eta x phi x layer``, and
  object-vector inputs, ``B x num_vectors x (pt, eta, phi, ...)``.
- YAML-driven model construction and compilation via ``wp21_ml_utils.model``.
- Base classes for constructing paired training/validation datasets and
  recording custom scalar objectives during Keras training.
- Keras serialisation support for the package's custom layers, losses, and
  regularisers.

Installation
------------

The package is available via PyPI:

.. code-block:: bash

   pip install wp21_ml_utils

For development and testing, please clone and install locally via:

.. code-block:: bash

   pip install -e wp21_ml_utils[dev]

Dependencies
------------

- ``python>=3.10``
- ``tensorflow>=2.16``
- ``HGQ2>=0.1.8``
- ``pyyaml``

Core modules
------------

- ``model.py``: load YAML configs, register custom Keras objects, build and
  compile models from config dictionaries, and load saved Keras models.
- ``quantisers.py``: differentiable quantisation layers including
  ``QuadLinearQuantiser``, ``FlexibleQuantiser``, and ``EncodeCellEt``.
- ``converters.py``: conversions between sparse physics vectors and dense
  eta-phi images via ``VectorsToImage`` and ``ImageToVectors``.
- ``pileup.py``: pileup-suppression layers including ``TowerSoftKiller`` and
  ``PileupCNN``.
- ``clustering.py``: cone-based jet reconstruction with ``ConeJet``.
- ``calibration.py``: transverse-momentum calibration with
  ``CalibrationMLP``.
- ``sequential.py``: reusable callable dense and convolutional-pooling network
  blocks that can be referenced from YAML configurations.
- ``layers.py``: reusable Keras layer components such as eta-phi padding,
  symmetry-aware convolutions, local-maximum masks, vector sums, and monotonic
  dense layers.
- ``losses.py``: custom training objectives including Chamfer, sparsity,
  calibration, and pinball losses.
- ``regularisers.py``: custom Keras regularisers for weight normalisation and
  sparsity.
- ``data.py``: ``BaseDataset`` interface for constructing batched training and
  validation datasets.
- ``callbacks.py``: ``BaseObjective`` callback for recording user-defined
  scalar scores in the Keras epoch logs.
- ``utils.py``: numerical helpers for momenta, coordinates, medians, image
  augmentation, and layer initialisation.

Datasets and custom objectives
------------------------------

Subclass ``BaseDataset`` to return batched training and validation datasets:

.. code-block:: python

   import numpy as np
   import tensorflow as tf
   from wp21_ml_utils.data import BaseDataset

   class NpzDataset(BaseDataset):
       def __init__(self, path):
           self.path = path

       def prepare_datasets(self):
           with np.load(self.path) as data:
               x, y = data["X"], data["y"]
           split = int(0.8 * len(x))
           make_ds = lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))
           return make_ds(x[:split], y[:split]), make_ds(x[split:], y[split:])

   train_ds, valid_ds = NpzDataset("my_data.npz")()

Subclass ``BaseObjective`` to add a scalar score to the epoch logs under its
``name``:

.. code-block:: python

   from wp21_ml_utils.callbacks import BaseObjective

   class ValidationMSE(BaseObjective):
       def __init__(self, x_valid, y_valid):
           super().__init__(name="validation_mse")
           self.x_valid = x_valid
           self.y_valid = y_valid

       def scoring_function(self):
           y_pred = self.model.predict(self.x_valid, verbose=0)
           return tf.reduce_mean(tf.square(self.y_valid - y_pred))

   model.fit(X, y, callbacks=[ValidationMSE(x_valid, y_valid)])

Config-driven model building
----------------------------

Model graphs can be described in YAML. The top-level sections are:

- ``inputs``: named Keras inputs with their tensor shapes.
- ``layers``: ordered computation nodes. Each node has a ``class``, one or
  more ``inputs``, and optional ``params`` passed to the constructor. The
  ``class`` may be a Keras ``Layer`` (including an HGQ2 layer) or a callable
  class that builds a reusable computation block.
- ``outputs``: named tensors to expose as model outputs, with optional loss,
  metrics, and loss-weight settings used by ``compile_from_config``.
- ``optimiser``: optional Keras optimiser name plus constructor parameters.
  Defaults to ``adam``.
- ``hgq_config``: optional HGQ2 quantizer and layer scopes applied by
  ``build_from_config`` while constructing the model. If not provided, no HGQ2
  scopes are applied.
- ``random_state``: optional random seed used during model construction.
  Defaults to 42.

Example configuration:

.. code-block:: yaml

   inputs:
     features:
       shape: [50, 64, 6]

   layers:
     conv_block:
       class: Conv2DPoolingLayers
       inputs: [features]
       params:
         filter_sizes: [4, 6, 8]
         kernel_sizes: 3
         pooling_sizes: 2
         activation: relu
         pooling: max
         use_hgq: true

     flatten:
       class: Flatten
       inputs: [conv_block]

     logits:
       class: QDense
       inputs: [flatten]
       params:
         units: 1

   outputs:
     logits:
       loss: BinaryCrossentropy
       params:
         from_logits: true
       metrics:
         - accuracy

   optimiser:
     class: adam
     params:
       learning_rate: 0.001
       clipnorm: 1.

   hgq_config:
     quantizer_scopes:
       - place: all
         default_q_type: kbi
         overflow_mode: SAT_SYM
         heterogeneous_axis: []
       - q_type: kbi
         place: [weight, bias]
         b0: 8
         i0: 2
       - place: datalane
         default_q_type: kif
         overflow_mode: WRAP
         f0: 6
     layer:
       enable_ebops: true
       beta0: 1.0e-6

   random_state: 42


Build and compile the model from the configuration as follows:

.. code-block:: python

   from wp21_ml_utils.model import (
       load_config,
       build_from_config,
       compile_from_config,
   )

   config = load_config("model_config.yaml")
   model, layers, tensors = build_from_config(config)
   compile_from_config(model, config)

The ``class`` values in the YAML are resolved through either registered custom
objects or standard Keras layers.

Callable classes
~~~~~~~~~~~~~~~~

The model builder accepts both standard Keras ``Layer`` subclasses and
callable Python classes. Callable classes encapsulate reusable computation
blocks composed of multiple Keras layers. They are constructed once from the
YAML parameters and invoked on their inputs during model construction.

For example, ``DenseLayers`` and ``Conv2DPoolingLayers`` in
``sequential.py`` are callable classes rather than Keras layers. When used in
a configuration they expand into the corresponding sequence of Keras layers,
rather than appearing as a single layer in the final model.

HGQ2 configuration
~~~~~~~~~~~~~~~~~~

``build_from_config`` can create the HGQ2 configuration scopes needed for
quantisation-aware training. Add an optional ``hgq_config`` mapping containing:

- ``quantizer_scopes``: a list of keyword-argument mappings passed to
  ``hgq.config.QuantizerConfigScope``. Scopes are entered in list order, so a
  later, more specific scope can refine an earlier general scope.
- ``layer``: a keyword-argument mapping passed to
  ``hgq.config.LayerConfigScope``.

Both kinds of scope remain active while every configured layer is constructed.
The complete example above sets general quantiser defaults, overrides the
weight and bias precision, configures datalane quantisation, and enables EBOPs
on HGQ2 layers. If ``hgq_config`` is absent, the builder creates no HGQ2 scope.

Extending the package with custom layers
----------------------------------------

User-defined custom layers and callable computation blocks can be registered
with ``update_custom_objects``.

.. code-block:: python

   from tensorflow.keras.layers import Layer
   from wp21_ml_utils.model import update_custom_objects

   class MyCustomLayer(Layer):
       def call(self, inputs):
           return inputs * 2

   update_custom_objects({"MyCustomLayer": MyCustomLayer})

Registering the object before calling ``build_from_config`` allows it to be
referenced in the model configuration, whether it is a Keras ``Layer`` or a
callable class.

License
-------

GNU Lesser General Public License v3 (LGPLv3)
