from typing import Callable
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw
import numpy as np

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
        k: int,
        target_column: str = 'target'
    ):
        self.k = k
        self.target_column = target_column
        
    def compute_knn(
        self,
        a,
        b,
        distance_expression,
        return_predictions: bool = True
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
        return_predictions : bool, default=True
            If True, return predicted classes (mode of k neighbors).
            If False, return grouped DataFrame with all k neighbors.
        
        Returns
        -------
        nw.Series or nw.DataFrame
            If return_predictions=True: Series with predicted class for each row in a.
            If return_predictions=False: Grouped DataFrame with k nearest neighbors 
            for each row in a, grouped by the original row index.
        """
        a = nw.from_native(a)

        index_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_row_index(index_column)
        
        # Convert b to same backend as a for cross join compatibility
        b_nw = nw.from_native(b)
        # Convert b's native backend to match a's native backend
        if a.to_native().__class__.__name__ != b_nw.to_native().__class__.__name__:
            # Different backends - need to convert b to a's backend
            import pandas as pd
            import polars as pl
            b_native = b_nw.to_native()
            a_native = a.to_native()
            
            if isinstance(a_native, pd.DataFrame) and not isinstance(b_native, pd.DataFrame):
                # Convert b to pandas
                b = nw.from_native(b_native.to_pandas())
            elif isinstance(a_native, pl.DataFrame) and not isinstance(b_native, pl.DataFrame):
                # Convert b to polars
                b = nw.from_native(pl.from_pandas(b_native))
            else:
                b = b_nw
        else:
            b = b_nw
        
        a = a.join(b, how='cross')
        
        # Add concatenated compression length column after cross join
        # This must be done here because we need columns from both a and b
        def compress_concat_to_numpy(s):
            """Compress concatenated values and return lengths as numpy array."""
            compressed_lengths = [len(self.compressor(x.encode(self.encoding) if isinstance(x, str) else x)) for x in s]
            return np.array(compressed_lengths, dtype=np.int64)
        
        a = a.with_columns(
            (nw.col(self._a_data_name) + nw.col(self.data_column))
            .map_batches(compress_concat_to_numpy)
            .cast(nw.Int64)
            .alias('_concat_compressed_len')
        )

        distance_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_columns((distance_expression).alias(distance_column))

        rank_column = nw.generate_temporary_column_name(6, columns=a.columns)
        a = a.with_columns(
            nw.col(distance_column).rank().over(index_column, order_by=distance_column).alias(rank_column)
        )

        a = a.filter(nw.col(rank_column) <= self.k)

        if return_predictions:
            # Aggregate to get predicted class (mode of k neighbors)
            result = a.group_by(index_column).agg(
                nw.col(self.target_column).mode().first()
            )
            # Sort by index to maintain original order
            result = result.sort(index_column)
            # Return just the predictions as a Series
            return result[self.target_column]
        else:
            # Return grouped DataFrame for inspection
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
        encoded=False,
        target_column: str = 'target'
    ):
        self.encoding = encoding
        self.encoded = encoded
        self.data_column = data_column

        if type(compressor) is str:
            self.compressor = self._get_callable_compressor(compressor)

        elif callable(compressor):
            self.compressor = compressor
        
        super().__init__(k, target_column=target_column)
        
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
        
        return self

    def predict(
        self,
        X: nw.DataFrame | nw.Series
    ):
        """Predict class labels for samples using compression distance.
        
        Computes Normalized Compression Distance (NCD) between each test sample
        and all training samples, then returns predicted class labels.
        
        Parameters
        ----------
        X : nw.DataFrame or nw.Series
            Test samples to classify.
        
        Returns
        -------
        nw.Series or nw.DataFrame

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
        # Build distance expression using only column references (no map_batches in expression)
        distance_expression = (
            (
                nw.col('_concat_compressed_len')
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

        preds = self.compute_knn(
            X, 
            self.model_data, 
            distance_expression=distance_expression, 
            return_predictions=True
        )

        return preds.to_native()

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
        def compress_to_series(s):
            """Compress series values and return lengths as numpy array."""
            compressed_lengths = [len(self.compressor(x.encode(self.encoding) if isinstance(x, str) else x)) for x in s]
            return np.array(compressed_lengths, dtype=np.int64)
        
        return nw.col(column).map_batches(compress_to_series).cast(nw.Int64)
    
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
        if isinstance(y, nw.DataFrame):
            if len(y.columns) != 1:
                raise ValueError("y_train must have exactly one column.")

            else:
                self.target_column = y.columns[0]
        
        elif isinstance(y, nw.Series):
            # Check if Series has a name
            if y.name is None:
                # No name - use self.target_column and rename after converting to frame
                y = y.to_frame()
                y = y.rename({y.columns[0]: self.target_column})
            else:
                # Has a name - use it and update self.target_column
                y = y.to_frame()
                self.target_column = y.columns[0]

        
        self.model_data = nw.concat([self.model_data, y], how='horizontal')