from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, List, Tuple, TypeVar, Dict, Optional, Union, Any

import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike
from tensorflow.keras.layers import Dense
from tensorflow.types.experimental import TensorLike


T = TypeVar("T")


class MonoDense(Dense):
    """Monotonic counterpart of the regular Dense Layer of tf.keras

    This is an implementation of our Monotonic Dense Unit or Constrained Monotone Fully Connected Layer. The below is the figure from the paper for reference.

    - the parameter `monotonicity_indicator` corresponds to **t** in the figure below, and

    - parameters `is_convex`, `is_concave` and `activation_weights` are used to calculate the activation selector **s** as follows:

        - if `is_convex` or `is_concave` is **True**, then the activation selector **s** will be (`units`, 0, 0) and (0, `units`, 0), respecively.

        - if both  `is_convex` or `is_concave` is **False**, then the `activation_weights` represent ratios between $\\breve{s}$, $\\hat{s}$ and $\\tilde{s}$,
          respecively. E.g. if `activation_weights = (2, 2, 1)` and `units = 10`, then
    """

    def __init__(
        self,
        units: int,
        *,
        activation: Optional[Union[str, Callable[[TensorLike], TensorLike]]] = None,
        monotonicity_indicator: ArrayLike = 1,
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
        **kwargs: Any,
    ):
        """Constructs a new MonoDense instance.

        Params:
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use, it is assumed to be convex monotonically
                increasing function such as "relu" or "elu"
            monotonicity_indicator: Vector to indicate which of the inputs are monotonically increasing or
                monotonically decreasing or non-monotonic. Has value 1 for monotonically increasing,
                -1 for monotonically decreasing and 0 for non-monotonic.
            is_convex: convex if set to True
            is_concave: concave if set to True
            activation_weights: relative weights for each type of activation, the default is (1.0, 1.0, 1.0).
                Ignored if is_convex or is_concave is set to True
            **kwargs: passed as kwargs to the constructor of `Dense`

        Raise:
            ValueError:
                - if both **is_concave** and **is_convex** are set to **True**, or
                - if any component of activation_weights is negative or there is not exactly three components
        """
        if is_convex and is_concave:
            raise ValueError(
                "The model cannot be set to be both convex and concave (only linear functions are both)."
            )

        if len(activation_weights) != 3:
            raise ValueError(
                f"There must be exactly three components of activation_weights, but we have this instead: {activation_weights}."
            )

        if (np.array(activation_weights) < 0).any():
            raise ValueError(
                f"Values of activation_weights must be non-negative, but we have this instead: {activation_weights}."
            )

        super(MonoDense, self).__init__(units=units, activation=None, **kwargs)

        self.units = units
        self.org_activation = activation
        self.monotonicity_indicator = monotonicity_indicator
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = activation_weights

        (
            self.convex_activation,
            self.concave_activation,
            self.saturated_activation,
        ) = self.get_activation_functions(self.org_activation)

    def get_saturated_activation(
        self,
        convex_activation: Callable[[TensorLike], TensorLike],
        concave_activation: Callable[[TensorLike], TensorLike],
        a: float = 1.0,
        c: float = 1.0,
    ) -> Callable[[TensorLike], TensorLike]:
        @tf.function
        def saturated_activation(
            x: TensorLike,
            convex_activation: Callable[[TensorLike], TensorLike] = convex_activation,
            concave_activation: Callable[[TensorLike], TensorLike] = concave_activation,
            a: float = a,
            c: float = c,
        ) -> TensorLike:
            cc = convex_activation(tf.ones_like(x) * c)
            return a * tf.where(
                x <= 0,
                convex_activation(x + c) - cc,
                concave_activation(x - c) + cc,
            )

        return saturated_activation  # type: ignore

    @lru_cache
    def get_activation_functions(
        self,
        activation: Optional[Union[str, Callable[[TensorLike], TensorLike]]] = None,
    ) -> Tuple[
        Callable[[TensorLike], TensorLike],
        Callable[[TensorLike], TensorLike],
        Callable[[TensorLike], TensorLike],
    ]:
        convex_activation = tf.keras.activations.get(
            activation.lower() if isinstance(activation, str) else activation
        )

        @tf.function
        def concave_activation(x: TensorLike) -> TensorLike:
            return -convex_activation(-x)

        saturated_activation = self.get_saturated_activation(
            convex_activation, concave_activation
        )
        return convex_activation, concave_activation, saturated_activation

    @tf.function
    def apply_activations(
        self,
        x: TensorLike,
        *,
        units: int,
        convex_activation: Callable[[TensorLike], TensorLike],
        concave_activation: Callable[[TensorLike], TensorLike],
        saturated_activation: Callable[[TensorLike], TensorLike],
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
    ) -> TensorLike:
        if convex_activation is None:
            return x

        elif is_convex:
            normalized_activation_weights = np.array([1.0, 0.0, 0.0])
        elif is_concave:
            normalized_activation_weights = np.array([0.0, 1.0, 0.0])
        else:
            if len(activation_weights) != 3:
                raise ValueError(f"activation_weights={activation_weights}")
            if (np.array(activation_weights) < 0).any():
                raise ValueError(f"activation_weights={activation_weights}")
            normalized_activation_weights = np.array(activation_weights) / sum(
                activation_weights
            )

        s_convex = round(normalized_activation_weights[0] * units)
        s_concave = round(normalized_activation_weights[1] * units)
        s_saturated = units - s_convex - s_concave

        x_convex, x_concave, x_saturated = tf.split(
            x, (s_convex, s_concave, s_saturated), axis=-1
        )

        y_convex = convex_activation(x_convex)
        y_concave = concave_activation(x_concave)
        y_saturated = saturated_activation(x_saturated)

        y = tf.concat([y_convex, y_concave, y_saturated], axis=-1)

        return y

    def get_monotonicity_indicator(
        self,
        monotonicity_indicator: ArrayLike,
        *,
        input_shape: Tuple[int, ...],
        units: int,
    ) -> TensorLike:
        # convert to tensor if needed and make it broadcastable to the kernel
        monotonicity_indicator = np.array(monotonicity_indicator)
        if len(monotonicity_indicator.shape) < 2:
            monotonicity_indicator = np.reshape(monotonicity_indicator, (-1, 1))
        elif len(monotonicity_indicator.shape) > 2:
            raise ValueError(
                f"monotonicity_indicator has rank greater than 2: {monotonicity_indicator.shape}"
            )

        if not np.all(
            (monotonicity_indicator == -1)
            | (monotonicity_indicator == 0)
            | (monotonicity_indicator == 1)
        ):
            raise ValueError(
                f"Each element of monotonicity_indicator must be one of -1, 0, 1, but it is: '{monotonicity_indicator}'"
            )
        return monotonicity_indicator

    def apply_monotonicity_indicator_to_kernel(
        self,
        kernel: tf.Variable,
        monotonicity_indicator: ArrayLike,
    ) -> TensorLike:
        # convert to tensor if needed and make it broadcastable to the kernel
        monotonicity_indicator = tf.convert_to_tensor(monotonicity_indicator)

        # absolute value of the kernel
        abs_kernel = tf.abs(kernel)

        # replace original kernel values for positive or negative ones where needed
        xs = tf.where(
            monotonicity_indicator == 1,
            abs_kernel,
            kernel,
        )
        xs = tf.where(monotonicity_indicator == -1, -abs_kernel, xs)

        return xs

    @contextmanager
    def replace_kernel_using_monotonicity_indicator(
        self,
        layer: tf.keras.layers.Dense,
        monotonicity_indicator: TensorLike,
    ):
        old_kernel = layer.kernel.value

        modified_kernel = self.apply_monotonicity_indicator_to_kernel(
            layer.kernel, monotonicity_indicator
        )

        layer.kernel.assign(modified_kernel)

        try:
            yield
        finally:
            layer.kernel.assign(old_kernel)

    def get_config(self) -> Dict[str, Any]:
        """Get config is used for saving the model"""
        return dict(
            units=self.units,
            activation=self.org_activation,
            monotonicity_indicator=self.monotonicity_indicator,
            is_convex=self.is_convex,
            is_concave=self.is_concave,
            activation_weights=self.activation_weights,
        )

    def build(self, input_shape: Tuple, *args: List[Any], **kwargs: Any) -> None:
        """Build

        Args:
            input_shape: input tensor
            args: positional arguments passed to Dense.build()
            kwargs: keyword arguments passed to Dense.build()
        """
        super(MonoDense, self).build(input_shape, *args, **kwargs)
        self.monotonicity_indicator = self.get_monotonicity_indicator(
            monotonicity_indicator=self.monotonicity_indicator,
            input_shape=input_shape,
            units=self.units,
        )

    def call(self, inputs: TensorLike) -> TensorLike:
        """Call

        Args:
            inputs: input tensor of shape (batch_size, ..., x_length)

        Returns:
            N-D tensor with shape: `(batch_size, ..., units)`.

        """
        # calculate W'*x+y after we replace the kernal according to monotonicity vector
        with self.replace_kernel_using_monotonicity_indicator(
            self, monotonicity_indicator=self.monotonicity_indicator
        ):
            h = super(MonoDense, self).call(inputs)

        y = self.apply_activations(
            h,
            units=self.units,
            convex_activation=self.convex_activation,
            concave_activation=self.concave_activation,
            saturated_activation=self.saturated_activation,
            is_convex=self.is_convex,
            is_concave=self.is_concave,
            activation_weights=self.activation_weights,
        )

        return y

    def _prepare_mono_input_n_param(
        self,
        inputs: Union[TensorLike, Dict[str, TensorLike], List[TensorLike]],
        param: Union[T, Dict[str, T], List[T]],
    ) -> Tuple[List[TensorLike], List[T], List[str]]:
        if isinstance(inputs, list):
            if isinstance(param, int):
                param = [param] * len(inputs)  # type: ignore
            elif isinstance(param, list):
                if len(inputs) != len(param):
                    raise ValueError(f"{len(inputs)} != {len(param)}")
            else:
                raise ValueError(f"Uncompatible types: {type(inputs)=}, {type(param)=}")
            sorted_feature_names = [f"{i}" for i in range(len(inputs))]

        elif isinstance(inputs, dict):
            sorted_feature_names = sorted(inputs.keys())

            if isinstance(param, int):
                param = [param] * len(inputs)  # type: ignore
            elif isinstance(param, dict):
                if set(param.keys()) != set(sorted_feature_names):
                    raise ValueError(
                        f"{set(param.keys())} != {set(sorted_feature_names)}"
                    )
                else:
                    param = [param[k] for k in sorted_feature_names]
            else:
                raise ValueError(f"Uncompatible types: {type(inputs)=}, {type(param)=}")

            inputs = [inputs[k] for k in sorted_feature_names]

        else:
            if not isinstance(param, int):
                raise ValueError(f"Uncompatible types: {type(inputs)=}, {type(param)=}")
            inputs = [inputs]
            param = [param]  # type: ignore
            sorted_feature_names = ["inputs"]

        return inputs, param, sorted_feature_names

    def _check_convexity_params(
        self,
        monotonicity_indicator: List[int],
        is_convex: List[bool],
        is_concave: List[bool],
        names: List[str],
    ) -> Tuple[bool, bool]:
        ix = [
            i
            for i in range(len(monotonicity_indicator))
            if is_convex[i] and is_concave[i]
        ]

        if len(ix) > 0:
            raise ValueError(
                f"Parameters both convex and concave: {[names[i] for i in ix]}"
            )

        has_convex = any(is_convex)
        has_concave = any(is_concave)
        if has_convex and has_concave:
            print("WARNING: we have both convex and concave parameters")

        return has_convex, has_concave
