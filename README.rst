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

Or to additionally install TensorFlow, Keras and HGQ2:

.. code-block:: bash

   pip install -e .[tf]

For development and testing:

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
- ``algorithms``: ordered layer nodes. Each node has an ``op``, one or more
  ``inputs``, and optional ``params`` passed to the layer constructor.
- ``outputs``: named tensors to expose as model outputs, with optional loss
  and loss-weight settings used by ``compile_from_config``.
- ``optimiser``: a Keras optimiser name plus constructor parameters.
- ``random_state``: TensorFlow seed used during model construction.

Example configuration:

.. code-block:: yaml

   inputs:
     cells:
       shape: [null, 4]  # pt, eta, phi, layer

   algorithms:
     encode_cells:
       op: EncodeCellEt
       inputs: cells
       params:
         encoding:
           class_name: QuadLinearQuantiser
           config:
             bits: 6
             trainable: true

     towers:
       op: VectorsToImage
       inputs: encode_cells
       params:
         return_layers: true
         filter_layers: [0, 1, 2, 3, 4, 5]

     pileup:
       op: PileupCNN
       inputs: towers

     jets:
       op: ConeJet
       inputs: pileup
       params:
         max_jets: 20

     calib:
       op: CalibrationMLP
       inputs: jets

     leading_pt:
       op: NthLeadingPt
       inputs: calib
       params:
         index: 1

   outputs:
     leading_pt:
       loss: mse
     calib:
       loss: CalibrationLoss
       loss_weight: 0.5

   optimiser:
     op: adam
     params:
       learning_rate: 0.001
       clipnorm: 1.0

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

The ``op`` values in the YAML are resolved through Keras custom objects.
``build_from_config`` registers classes from the package modules before
deserialising each layer.

License
-------

BSD 2-clause
