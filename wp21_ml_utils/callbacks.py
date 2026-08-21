from tensorflow.keras.callbacks import Callback


class BaseObjective(Callback):
    """Base callback for recording a custom scalar objective after each epoch.

    Subclasses must implement ``scoring_function``. At the end of each epoch,
    the computed score is added to the Keras logs dictionary under ``name``.

    Parameters
    ----------
    name : str
        Key under which the objective value is stored in the Keras logs.
    """

    def __init__(self, name: str):
        """Initialise the objective callback.

        Parameters
        ----------
        name : str
            Key under which the objective value is stored in the Keras logs.
        """
        super().__init__()
        self.name = name

    def scoring_function(self) -> float:
        """Compute the custom objective value.

        Subclasses must override this method and return a scalar value that
        can be converted to a Python ``float``.

        Returns
        -------
        float
            A scalar custom objective value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by a subclass.
        """
        raise NotImplementedError("please overload the `scoring_function` method.")

    def on_epoch_end(self, epoch, logs=None):
        """Compute the objective and add it to the epoch logs.

        Parameters
        ----------
        epoch : int
            Index of the epoch that has ended.
        logs : dict or None, optional
            Dictionary of metrics collected by Keras. If ``None``, the score
            is not computed.
        """
        if logs is not None:
            logs[self.name] = self.score()

    def score(self) -> float:
        """Compute and validate the custom objective value.

        Returns
        -------
        float
            The value returned by ``scoring_function``, converted to a Python
            ``float``.

        Raises
        ------
        ValueError
            If the result of ``scoring_function`` cannot be converted to a
            Python ``float``.
        """
        score = self.scoring_function()
        try:
            return float(score)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "`scoring_function` must return a scalar value."
            ) from error
