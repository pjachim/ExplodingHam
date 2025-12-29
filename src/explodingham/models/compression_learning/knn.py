from typing import Callable
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw

class BaseKNNModel(BaseExplodingHamClassifier):
    def __init__(
        self,
        k: int
    ):
        self.k = k
        
    def compute_knn(
        self,
        a,
        b,
        distance_expression
    ):
        """Compute pairwise distance matrix between two DataFrames."""
        a = nw.from_native(a)

        index_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_row_index(index_column)
        b = nw.from_native(b)
        a = a.join(b, how='cross')

        distance_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_columns((distance_expression).alias(distance_column))

        rank_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_columns(
            nw.col(distance_column).rank().over(index_column, order_by=distance_column).alias(rank_column)
        )

        a = a.filter(nw.col(rank_column) <= self.k)

        a = a.group_by(index_column)

        return a
    
class CompressionKNN(BaseKNNModel):
    def __init__(
        self,
        k: int,
        data_column: str | None = None,
        encoding: str = 'utf-8',
        compressor: str | Callable = 'gzip',
        encoded=False
    ):
        self.encoding = encoding
        self.encoded = encoded
        self.data_column = data_column

        if type(compressor) is str:
            self.compressor = self._get_callable_compressor(compressor)

        elif callable(compressor):
            self.compressor = compressor
        
        super().__init__(k)
        
    def fit(
        self,
        X_train: nw.DataFrame | nw.Series,
        y_train: nw.DataFrame | nw.Series
    ):
        self.model_data = self._handle_X(X_train)
        self.backend = self.model_data.implementation

        y = nw.from_native(y_train, allow_series=True)
        # Make model_data contain both X and y
        self._add_y_to_X(y)

        # Add compressed length column to model_data to reduce redundant computations
        self._compressed_a_len_name = nw.generate_temporary_column_name(6, columns=self.model_data.columns)
        self._a_data_name = nw.generate_temporary_column_name(6, columns=list(self.model_data.columns) + [self._compressed_a_len_name])
        self.model_data = self.model_data.with_columns(
            self._get_compressed_len(self.data_column).alias(self._compressed_a_len_name),
            nw.col(self.data_column).alias(self._a_data_name)
        )

        self.model_data = self.model_data.select([
            self._a_data_name,
            self._compressed_a_len_name,
            self.target_column
        ])

    def predict(
        self,
        X: nw.DataFrame | nw.Series
    ):
        X = self._handle_X(X)

        # Add compressed length column to X to reduce computations
        compressed_b_len_name = nw.generate_temporary_column_name(6, columns=list(self.model_data.columns) + list(X.columns))
        X = X.with_columns(
            self._get_compressed_len(self.data_column).alias(compressed_b_len_name)
        )

        # Per https://aclanthology.org/2023.findings-acl.426.pdf
        # NCD(x, y) = [C(xy) − min{C(x), C(y)}] / max{C(x), C(y)}
        distance_expression = (
            (
                self._get_compressed_len(self._a_data_name + nw.col(self.data_column))
                - nw.min_horizontal(
                    nw.col(self._compressed_a_len_name),
                    nw.col(compressed_b_len_name)
                )
            )
            / nw.max_horizontal(
                nw.col(compressed_b_len_name),
                nw.col(self._compressed_a_len_name)
            )
        )

        return self.compute_knn(X, self.model_data, distance_expression=distance_expression)

    def _get_compressed_len(self, column: str) -> nw.Expr:
        if self.encoded:
            col_expr = nw.col(column)
        else:
            col_expr = self._encode(column)

        return nw.col(column).map_batches(
                lambda s: nw.new_series(name=column, values=[self.compressor(x) for x in s], backend=self.backend).to_native(),
                return_dtype=nw.Binary
            ).len().alias(column)
    
    def _encode(self, column: str) -> nw.Expr:
        return nw.col(column).map_batches(
                lambda s: nw.new_series(name=column, values=[lambda x: x.encode(self.encoding) for x in s], backend=self.backend).to_native(),
                return_dtype=nw.Binary
            ).len().alias(column)
    
    def _encode_string(self, string: str) -> nw.Expr:
        return string.encode(self.encoding)
    
    def _compress_bytes(self, byte_data: bytes) -> bytes:
        return self.compressor(byte_data)
    
    def _handle_X(self, X: nw.DataFrame | nw.Series) -> nw.DataFrame:
        X = nw.from_native(X, allow_series=True, eager_only=True)
        
        if isinstance(X, nw.Series):
            X = X.to_frame()
        
        return X
        

    def _get_callable_compressor(self, compressor_name: str) -> callable:
        if compressor_name == 'gzip':
            import gzip
            return gzip.compress
        
        elif compressor_name == 'bz2':
            import bz2
            return bz2.compress
        
        elif compressor_name == 'lzma':
            import lzma
            return lzma.compress
        
        else:
            raise ValueError(f"Unsupported compressor: {compressor_name}")
        
    def _add_y_to_X(self, y: nw.DataFrame | nw.Series) -> None:
        if type(y) is nw.DataFrame:
            if len(y.columns) != 1:
                raise ValueError("y_train must have exactly one column.")

            else:
                self.target_column = y.columns[0]
        
        elif type(y) is nw.Series:
            self.target_column = y.name
            y = y.to_frame()

        
        self.model_data = nw.concat([self.model_data, y], how='horizontal')

#class CompressionKNN(BaseExplodingHamClassifier):
#    def __init__(
#            self,
#            n_neighbors: int = 5,
#            compression: str = 'gzip',
#            token_scrubbing: bool = False,
#            encoding: str = 'utf-8'
#        ) -> None:
#        """
#        K-Nearest Neighbors classifier using compression-based distance metrics.
#        Parameters
#        ----------
#        n_neighbors : int, optional
#            Number of neighbors to use (default is 5).
#        compression : str, optional
#            Compression method to use (default is 'gzip').
#        token_scrubbing : bool, optional
#            Whether to use token scrubbing (default is False).
#        encoding : str, optional
#            Encoding to use when converting strings to bytes (default is 'utf-8').
#        """
#        self._set_param('n_neighbors', n_neighbors, int)
#        self._set_option('compression', compression, ['gzip'])
#        self._set_param('token_scrubbing', token_scrubbing, bool)
#        self._set_param('encoding', encoding, str)
#        self.ncd = NormalizedCompressionDistance(
#            compressor = self._get_compression_function()
#        )
#
#    def _get_compression_function(self) -> callable:
#        """
#        Get the compression function based on the specified method.
#        Parameters
#        ----------
#        method : str
#            Compression method name.
#        Returns
#        -------
#        compressor : Callable
#            Compression function.
#        """
#        if self.compression == 'gzip':
#            import gzip
#            return gzip.compress
#        
#        raise NotImplementedError(f"Compression method '{self.compression}' is not implemented.")
#
#    def _convert_to_bytes(self, x: str | bytes) -> bytes:
#        """
#        Convert input to bytes if it's a string.
#        Parameters
#        ----------
#        x : str | bytes
#            Input data.
#        Returns
#        -------
#        bytes
#            Input data as bytes.
#        """
#        if isinstance(x, str):
#            return x.encode(self.encoding)
#        return x
#
#    def fit(self, X, y):
#        """
#        Fit the CompressionKNN model.
#        Parameters
#        ----------
#        X : array-like, shape (n_samples, n_features)
#            Training data (strings or bytes).
#        y : array-like, shape (n_samples,)
#            Target labels.
#        Returns
#        -------
#        self : object
#            Fitted estimator.
#        """
#        # Convert all training data to bytes
#        self.stored_X = [self._convert_to_bytes(x) for x in X]
#        self.stored_y = y
#
#        return self
#    
#    def predict(self, X) -> list:
#        """
#        Predict the class labels for the provided data.
#        Parameters
#        ----------
#        X : array-like, shape (n_samples, n_features)
#            Input data.
#        Returns
#        -------
#        y_pred : array, shape (n_samples,)
#            Predicted class labels.
#        """
#        return self._predict_labels_or_predict_proba(X, return_probs=False)
#    
#    def predict_proba(self, X) -> list:
#        """
#        Predict class probabilities for the provided data.
#        Parameters
#        ----------
#        X : array-like, shape (n_samples, n_features)
#            Input data.
#        Returns
#        -------
#        y_proba : array, shape (n_samples,)
#            Predicted class probabilities.
#        """
#        return self._predict_labels_or_predict_proba(X, return_probs=True)
#
#    def _predict_labels_or_predict_proba(self, X, return_probs: bool) -> list:
#        labels = []
#        for x in X:
#            # Convert test sample to bytes
#            x_bytes = self._convert_to_bytes(x)
#            distances = [
#                self.ncd.ncd(x_bytes, train_x) for train_x in self.stored_X
#            ]
#
#            # Get top n_neighbors from self.stored_y based on distances
#            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
#            neighbor_labels = [self.stored_y[i] for i in neighbor_indices]
#
#            # Get label from mode of neighbor_labels (supports any hashable type)
#            label = Counter(neighbor_labels).most_common(1)[0][0]
#            if return_probs:
#                label = len([l for l in neighbor_labels if l == label]) / self.n_neighbors
#
#            labels.append(label)
#
#        return labels