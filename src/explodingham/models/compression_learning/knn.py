from explodingham.utils.distance_metrics import NormalizedCompressionDistance
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import numpy as np
from collections import Counter

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
        return self._predict_labels_or_predict_proba(X, return_probs=False)
    
    def predict_proba(self, X) -> list:
        """
        Predict class probabilities for the provided data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        y_proba : array, shape (n_samples,)
            Predicted class probabilities.
        """
        return self._predict_labels_or_predict_proba(X, return_probs=True)

    def _predict_labels_or_predict_proba(self, X, return_probs: bool) -> list:
        labels = []
        for x in X:
            distances = [
                self.ncd.ncd(x, train_x) for train_x in self.stored_X
            ]

            # Get top n_neighbors from self.stored_y based on distances
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = [self.stored_y[i] for i in neighbor_indices]

            # Get label from mode of neighbor_labels (supports any hashable type)
            label = Counter(neighbor_labels).most_common(1)[0][0]
            if return_probs:
                label = len([l for l in neighbor_labels if l == label]) / self.n_neighbors

            labels.append(label)

        return labels