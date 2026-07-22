wp21_ml_utils
=============

A compact TensorFlow/Keras utility package for HGQ-based ML studies on the
Global Trigger.

The package provides serialisable Keras layers, losses, regularisers, and
configuration helpers for building detector-inspired pipelines that operate on
both sparse object vectors and dense eta-phi images.

Features
--------

- Custom Keras layers for quantisation, eta-phi image conversion, pileup
  suppression, cone-jet reconstruction, and particle calibration.
- Support for image-style event inputs, ``B x eta x phi x layer``, and
  object-vector inputs, ``B x num_vectors x (pt, eta, phi, ...)``.
- YAML-driven model construction and compilation via
  ``wp21_ml_utils.model``.
- Keras serialisation support for the package's custom layers, losses, and
  regularisers.

Installation
------------

From the repository root:

.. code-block:: bash

   pip install -e .

For development and testing, additional dependencies can be installed via:

.. code-block:: bash

   pip install -e .[dev]

Dependencies
------------

- ``tensorflow[and-cuda]>=2.16``
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
- ``layers``: ordered layer nodes. Each node has an ``class``, one or more ``inputs``, and optional ``params`` passed to the layer constructor. Note that HGQ2 layer classes must be prefixed with ``hgq>``.
- ``outputs``: named tensors to expose as model outputs, with optional loss
  and loss-weight settings used by ``compile_from_config``.
- ``optimiser``: a Keras optimiser name plus constructor parameters.
- ``random_state``: TensorFlow seed used during model construction.

Example configuration:

.. code-block:: yaml

   inputs:
    cells:
      shape: [null, 4] # pt eta phi layer

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
        index: 1

    pt_4:
      class: NthLeadingPt
      inputs: [calib]
      params:
        index: 4

  outputs:
    pt_1:
      loss: mse
    pt_4:
      loss: MeanAbsoluteError
      loss_weight: 0.5
    calib:
      loss: CalibrationLoss

  optimiser:
    class: adam
    params:
      learning_rate: 0.001
      clipnorm: 1.

  random_state: 42


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

The ``layer`` values in the YAML are resolved through either Keras custom objects or standard Keras layers.
See the next section for how to register custom layers.

QAT can be enabled by calling ``build_from_config`` within the usual HGQ2 scope, for example:

.. code-block:: python

  from hgq.config import LayerConfigScope, QuantizerConfigScope

  with (
      QuantizerConfigScope(place="all", default_q_type="kbi", overflow_mode="SAT_SYM"),
      QuantizerConfigScope(place="datalane", default_q_type="kif", overflow_mode="WRAP"),
      LayerConfigScope(enable_ebops=True, beta0=1e-5),
    ):
    model, layers, tensors = build_from_config(config)

Extending the package with custom layers
----------------------------
User-defined custom layers can be registered with ``update_custom_objects``.

.. code-block:: python

  from wp21_ml_utils.model import update_custom_objects

  class MyCustomLayer(Layer):
    def call(self, inputs):
        return inputs * 2

  update_custom_objects({"MyCustomLayer": MyCustomLayer})

Registering the layer before calling ``build_from_config`` allows the layer to be referenced in the model configuration.

License
-------

BSD 2-clause
