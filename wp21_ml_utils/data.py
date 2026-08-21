import tensorflow as tf


class BaseDataset:
    """Base class for paired training and validation dataset factories.

    Subclasses must implement ``prepare_datasets`` and return batched training
    and validation datasets as a two-item tuple. Instances are callable as a
    convenient shorthand for that method.
    """

    def __call__(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare and return the training and validation datasets.

        Returns
        -------
        tuple[tf.data.Dataset, tf.data.Dataset]
            The batched training dataset followed by the batched validation
            dataset.
        """
        train_ds, valid_ds = self.prepare_datasets()
        return train_ds, valid_ds

    def prepare_datasets(self):
        """Construct the training and validation datasets.

        Subclasses must override this method.

        Returns
        -------
        tuple[tf.data.Dataset, tf.data.Dataset]
            The batched training dataset followed by the batched validation
            dataset.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by a subclass.
        """
        raise NotImplementedError()
