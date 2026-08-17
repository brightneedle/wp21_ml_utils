wp21_ml_utils
=============

A compact TensorFlow/Keras utility package for HGQ-based ML studies on the
Global Trigger.

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
- YAML-driven model construction and compilation via
  ``wp21_ml_utils.model``.
- Keras serialisation support for the package's custom layers, losses, and
  regularisers.

Installation
------------

The package is available via ``pypi``:

.. code-block:: bash

   pip install wp21_ml_utils

For development and testing, please clone and install locally via:

.. code-block:: bash

   pip install -e wp21_ml_utils[dev]

Dependencies
------------

- ``tensorflow>=2.16``
- ``HGQ2==0.1.8``
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
- ``utils.py``: numerical helpers for momenta, coordinates, medians, image
  augmentation, and layer initialisation.

Config-driven model building
----------------------------

Model graphs can be described in YAML. The top-level sections are:

- ``inputs``: named Keras inputs with their tensor shapes.
- ``layers``: ordered computation nodes. Each node has a ``class``, one or
  more ``inputs``, and optional ``params`` passed to the constructor. The
  ``class`` may be either a Keras ``Layer`` or a callable class that builds a
  reusable computation block. Note that HGQ2 layer classes must be prefixed
  with ``hgq>``.
- ``outputs``: named tensors to expose as model outputs, with optional loss,
  metrics, and loss-weight settings used by ``compile_from_config``.
- ``optimiser``: a Keras optimiser name plus constructor parameters.
- ``random_state``: TensorFlow seed used during model construction.

Example configuration:

.. code-block:: yaml

   inputs:
     cells:
       shape: [null, 4]  # var x (pt eta phi layer)

   layers:
     encode_cells:
       class: EncodeCellEt
       inputs: [cells]
       params:
         encoder_layer: QuadLinearQuantiser
         encoder_config:
           trainable: true

     towers:
       class: VectorsToImage
       inputs: [encode_cells]
       params:
         return_layers: true
         filter_layers: [0, 1, 2, 3, 4, 5]

     pileup:
       class: PileupCNN
       inputs: [towers]

     jets:
       class: ConeJet
       inputs: [pileup]

     calib:
       class: CalibrationMLP
       inputs: [jets]

     pt_1:
       class: NthLeadingPt
       inputs: [calib]
       params:
         index: 0

     pt_4:
       class: NthLeadingPt
       inputs: [calib]
       params:
         index: 3

   outputs:
     pt_1:
       loss: mse
       metrics:
         - mae
     pt_4:
       loss: MeanAbsoluteError
       loss_weight: 0.5
       metrics:
         - mse
     calib:
       loss: CalibrationLoss
       params:
        sqaured: false

   optimiser:
     class: adam
     params:
       learning_rate: 0.001
       clipnorm: 1.

   random_state: 42

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

Build and compile the model from that config:

.. code-block:: python

   from wp21_ml_utils.model import (
       load_config,
       build_from_config,
       compile_from_config,
       load_model,
   )

   config = load_config("model_config.yaml")
   model, layers, tensors = build_from_config(config)
   compile_from_config(model, config)

   model.save("pipeline.keras")
   restored = load_model("pipeline.keras")

The ``class`` values in the YAML are resolved through either registered custom
objects or standard Keras layers.

QAT can be enabled by calling ``build_from_config`` within the usual HGQ2
scope, for example:

.. code-block:: python

   from hgq.config import LayerConfigScope, QuantizerConfigScope

   with (
       QuantizerConfigScope(
           place="all",
           default_q_type="kbi",
           overflow_mode="SAT_SYM",
       ),
       QuantizerConfigScope(
           place="datalane",
           default_q_type="kif",
           overflow_mode="WRAP",
       ),
       LayerConfigScope(enable_ebops=True, beta0=1e-5),
   ):
       model, layers, tensors = build_from_config(config)

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

BSD 2-clause
```
