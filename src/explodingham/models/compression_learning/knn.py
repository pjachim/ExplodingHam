from typing import Callable
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw

class BaseKNNModel(BaseExplodingHamClassifier):
    """Base class for K-Nearest Neighbors models.
    
    This abstract base class provides core KNN functionality for finding
    k-nearest neighbors using a custom distance metric. Subclasses should
    implement specific distance metrics and prediction logic.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for predictions.
    
    Attributes
    ----------
    k : int
        Number of nearest neighbors.
    """
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
        """Compute k-nearest neighbors between two DataFrames.
        
        Performs a cross join between DataFrames a and b, computes distances
        using the provided expression, ranks neighbors by distance, and filters
        to keep only the k nearest neighbors for each row in a.
        
        Parameters
        ----------
        a : nw.DataFrame
            Query DataFrame for which to find nearest neighbors.
        b : nw.DataFrame
            Reference DataFrame containing candidate neighbors.
        distance_expression : nw.Expr
            Narwhals expression that computes distance between rows.
            Should reference columns from both a and b after cross join.
        
        Returns
        -------
        nw.DataFrame
            Grouped DataFrame with k nearest neighbors for each row in a,
            grouped by the original row index.
        """
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
    """K-Nearest Neighbors classifier using Normalized Compression Distance.
    
    This classifier uses compression-based similarity metrics to classify samples.
    It computes the Normalized Compression Distance (NCD) between samples, where
    NCD(x, y) = [C(xy) - min{C(x), C(y)}] / max{C(x), C(y)}, and C(·) is the
    compressed length. This approach leverages the principle that similar objects
    compress well together.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for classification.
    data_column : str or None, optional
        Name of the column containing data to compress. If None, will be inferred.
    encoding : str, default='utf-8'
        Character encoding to use when converting strings to bytes.
    compressor : str or Callable, default='gzip'
        Compression method. If string, must be one of 'gzip', 'bz2', or 'lzma'.
        If callable, should be a function that takes bytes and returns compressed bytes.
    encoded : bool, default=False
        If True, assumes input data is already encoded as bytes.
    
    Attributes
    ----------
    model_data : nw.DataFrame
        Training data with precomputed compressed lengths.
    target_column : str
        Name of the target variable column.
    compressor : Callable
        Compression function used for NCD computation.
    
    Notes
    -----
    The Normalized Compression Distance is based on Kolmogorov complexity theory
    and provides a universal similarity metric. The method is parameter-free
    (aside from k) and can work with any data type that can be serialized.
    
    References
    ----------
    .. [1] Cilibrasi, R., & Vitanyi, P. M. (2005). Clustering by compression.
           IEEE Transactions on Information theory, 51(4), 1523-1545.
    """
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
        """Fit the CompressionKNN classifier.
        
        Stores the training data and precomputes compressed lengths for efficiency.
        This avoids redundant compression operations during prediction.
        
        Parameters
        ----------
        X_train : nw.DataFrame or nw.Series
            Training data containing samples to compress.
        y_train : nw.DataFrame or nw.Series
            Target labels corresponding to X_train samples.
        
        Returns
        -------
        self : CompressionKNN
            Fitted classifier instance.
        """
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
        """Predict class labels for samples using compression distance.
        
        Computes Normalized Compression Distance (NCD) between each test sample
        and all training samples, then returns the k nearest neighbors.
        
        Parameters
        ----------
        X : nw.DataFrame or nw.Series
            Test samples to classify.
        
        Returns
        -------
        nw.DataFrame
            Grouped DataFrame containing k-nearest neighbors for each test sample.
            Groups are indexed by test sample row number.
        
        Notes
        -----
        The NCD formula used is:
        NCD(x, y) = [C(xy) - min{C(x), C(y)}] / max{C(x), C(y)}
        
        where C(·) denotes the compressed length of a sequence.
        """
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
        """Get compressed length of data in a column.
        
        Parameters
        ----------
        column : str
            Name of the column to compress.
        
        Returns
        -------
        nw.Expr
            Narwhals expression that computes compressed length for each value.
        """
        if self.encoded:
            # Data is already bytes, just compress
            return nw.col(column).map_batches(
                lambda s: nw.new_series(name=column, values=[self.compressor(x) for x in s], backend=self.backend).to_native(),
                return_dtype=nw.Object
            ).len()
        else:
            # Data is strings, encode AND compress in one operation
            return nw.col(column).map_batches(
                lambda s: nw.new_series(name=column, values=[self.compressor(x.encode(self.encoding)) for x in s], backend=self.backend).to_native(),
                return_dtype=nw.Object
            ).len()
    
    def _encode(self, column: str) -> nw.Expr:
        """Encode string data to bytes using configured encoding.
        
        Parameters
        ----------
        column : str
            Name of the column containing string data.
        
        Returns
        -------
        nw.Expr
            Narwhals expression that encodes strings to bytes.
        """
        return nw.col(column).map_batches(
                lambda s: nw.new_series(name=column, values=[x.encode(self.encoding) for x in s], backend=self.backend).to_native(),
                return_dtype=nw.Binary
            )
    
    def _encode_string(self, string: str) -> nw.Expr:
        """Encode a single string to bytes.
        
        Parameters
        ----------
        string : str
            String to encode.
        
        Returns
        -------
        bytes
            Encoded byte representation of the string.
        """
        return string.encode(self.encoding)
    
    def _compress_bytes(self, byte_data: bytes) -> bytes:
        """Compress byte data using configured compressor.
        
        Parameters
        ----------
        byte_data : bytes
            Data to compress.
        
        Returns
        -------
        bytes
            Compressed byte data.
        """
        return self.compressor(byte_data)
    
    def _handle_X(self, X: nw.DataFrame | nw.Series) -> nw.DataFrame:
        """Convert input data to DataFrame format.
        
        Parameters
        ----------
        X : nw.DataFrame or nw.Series
            Input data to convert.
        
        Returns
        -------
        nw.DataFrame
            Input data as a DataFrame.
        """
        X = nw.from_native(X, allow_series=True, eager_only=True)
        
        if isinstance(X, nw.Series):
            X = X.to_frame()
        
        return X
        

    def _get_callable_compressor(self, compressor_name: str) -> callable:
        """Get compression function from string identifier.
        
        Parameters
        ----------
        compressor_name : str
            Name of the compression algorithm. Must be one of:
            'gzip', 'bz2', or 'lzma'.
        
        Returns
        -------
        callable
            Compression function that takes bytes and returns compressed bytes.
        
        Raises
        ------
        ValueError
            If compressor_name is not supported.
        """
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
        """Add target variable to model data.
        
        Horizontally concatenates target labels with feature data and
        stores the target column name.
        
        Parameters
        ----------
        y : nw.DataFrame or nw.Series
            Target labels to add. If DataFrame, must have exactly one column.
        
        Raises
        ------
        ValueError
            If y is a DataFrame with more than one column.
        """
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