from explodingham.utils.distance_metrics import NormalizedCompressionDistance
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import numpy as np
from scipy.stats import mode

class CompressionKNN(BaseExplodingHamClassifier):
    def __init__(
            self,
            n_neighbors: int = 5,
            compression: str = 'gzip',
            token_scrubbing: bool = False
        ) -> None:
        """
        K-Nearest Neighbors classifier using compression-based distance metrics.
        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors to use (default is 5).
        compression : str, optional
            Compression method to use (default is 'gzip').
        token_scrubbing : bool, optional
            Whether to use token scrubbing (default is False).
        """
        self._set_param('n_neighbors', n_neighbors, int)
        self._set_option('compression', compression, ['gzip'])
        self._set_param('token_scrubbing', token_scrubbing, bool)
        self.ncd = NormalizedCompressionDistance(
            compressor = self._get_compression_function()
        )

    def _get_compression_function(self) -> callable:
        """
        Get the compression function based on the specified method.
        Parameters
        ----------
        method : str
            Compression method name.
        Returns
        -------
        compressor : Callable
            Compression function.
        """
        if self.compression == 'gzip':
            import gzip
            return gzip.compress
        
        raise NotImplementedError(f"Compression method '{self.compression}' is not implemented.")

    def fit(self, X, y):
        """
        Fit the CompressionKNN model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Implementation of fit method goes here
        self.stored_X = X
        self.stored_y = y

        return self
    
    def predict(self, X) -> list:
        """
        Predict the class labels for the provided data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        label = []
        for x in X:
            distances = [
                self.ncd.ncd(x, train_x) for train_x in self.stored_X
            ]

            # Get top n_neighbors from self.stored_y based on distances
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = [self.stored_y[i] for i in neighbor_indices]

            # Get label from mode of neighbor_labels
            most_common = mode(neighbor_labels).mode[0]
            label.append(most_common)

        return label