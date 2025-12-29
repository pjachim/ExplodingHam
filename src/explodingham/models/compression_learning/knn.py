from typing import Callable
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import narwhals as nw
import numpy as np

class BaseKNNModel(BaseExplodingHamClassifier):
    """Base class for K-Nearest Neighbors models using custom distance metrics.
    
    This abstract base class provides core KNN functionality for finding
    k-nearest neighbors using custom distance metrics. Unlike traditional KNN
    which relies on geometric distance measures (Euclidean, Manhattan, etc.),
    this base class supports arbitrary distance functions, enabling novel
    similarity measures such as compression-based distances.
    
    The class handles the mechanics of cross-joining query and reference data,
    computing distances, ranking neighbors, and extracting the k closest matches.
    Subclasses need only implement specific distance metrics and inherit the
    KNN computation pipeline.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for predictions. Must be positive.
        Larger values provide smoother decision boundaries but may include
        less relevant neighbors.
    target_column : str, default='target'
        Name of the column containing class labels in the training data.
    
    Attributes
    ----------
    k : int
        Number of nearest neighbors.
    target_column : str
        Name of the target variable column.
    
    Notes
    -----
    The KNN algorithm classifies samples by majority vote among the k nearest
    neighbors. The choice of k affects the bias-variance tradeoff: small k
    leads to low bias but high variance, while large k increases bias but
    reduces variance.
    
    This implementation uses Narwhals for dataframe-agnostic operations,
    enabling compatibility with both Pandas and Polars backends.
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
        """Compute k-nearest neighbors between two DataFrames using custom distance.
        
        This method performs a cross join between query data (a) and reference
        data (b), evaluates a distance expression for all pairs, ranks neighbors
        by distance, and filters to retain only the k nearest neighbors per query.
        
        The workflow is optimized for dataframe operations: cross join → compute
        distances → rank → filter → aggregate. This approach leverages vectorized
        operations rather than iterative distance calculations.
        
        Parameters
        ----------
        a : nw.DataFrame
            Query DataFrame for which to find nearest neighbors. Each row will
            be matched against all rows in b.
        b : nw.DataFrame
            Reference DataFrame containing candidate neighbors (typically the
            training data).
        distance_expression : nw.Expr
            Narwhals expression that computes distance between rows after cross
            join. Should reference columns from both a and b. Lower values indicate
            closer neighbors.
        return_predictions : bool, default=True
            If True, return predicted classes (mode of k neighbors' labels).
            If False, return grouped DataFrame with all k neighbors for inspection.
        
        Returns
        -------
        nw.Series or nw.DataFrame
            If return_predictions=True: Series with predicted class for each row
            in a, computed as the mode (most common) class among k neighbors.
            If return_predictions=False: Grouped DataFrame containing k nearest
            neighbors for each query row, indexed by the original row number.
        
        Notes
        -----
        The method handles backend compatibility by converting b to match a's
        dataframe backend (Pandas or Polars). This ensures cross join operations
        work correctly.
        
        Distance ties are broken arbitrarily by the ranking function. If multiple
        neighbors have identical distances at the k-th position, the selection
        may be non-deterministic.
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
    
    CompressionKNN leverages information theory and Kolmogorov complexity to
    classify data without explicit feature engineering. The classifier computes
    similarity between objects using compression: if two objects compress well
    together, they are considered similar. This makes it a universal similarity
    metric that works across diverse data types (text, DNA sequences, time series).
    
    The core insight is that compression algorithms approximate Kolmogorov
    complexity—the length of the shortest program that produces a given output.
    Objects with similar structure share algorithmic information, causing their
    concatenation to compress better than the sum of their individual compressions.
    
    The Normalized Compression Distance (NCD) quantifies this:
    
    .. math::
        NCD(x, y) = \\frac{C(xy) - \\min\\{C(x), C(y)\\}}{\\max\\{C(x), C(y)\\}}
    
    where C(·) is the compressed length and xy denotes concatenation. NCD ranges
    from 0 (identical) to 1+ (dissimilar).
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for classification. Must be positive.
        Odd values are preferred for binary classification to avoid ties.
    data_column : str or None, optional
        Name of the column containing data to compress. If None, will be inferred
        from the first non-target column.
    encoding : str, default='utf-8'
        Character encoding to use when converting strings to bytes before
        compression. Common options: 'utf-8', 'ascii', 'latin-1'.
    compressor : str or Callable, default='gzip'
        Compression method to use for computing NCD. If string, must be one of:
        
        - 'gzip': Fast, general-purpose (good default)
        - 'bz2': Better compression ratio, slower
        - 'lzma': Best compression ratio, slowest
        
        If callable, should be a function with signature `f(bytes) -> bytes`.
    encoded : bool, default=False
        If True, assumes input data is already encoded as bytes and skips
        encoding step.
    target_column : str, default='target'
        Name of the column containing class labels in training data.
    
    Attributes
    ----------
    model_data : nw.DataFrame
        Training data with precomputed compressed lengths for efficiency.
    target_column : str
        Name of the target variable column.
    compressor : Callable
        Compression function used for NCD computation.
    encoding : str
        Character encoding for string-to-bytes conversion.
    k : int
        Number of nearest neighbors.
    
    Examples
    --------
    Basic text classification by language:
    
    >>> import pandas as pd
    >>> from explodingham.models.compression_learning.knn import CompressionKNN
    >>> 
    >>> X_train = pd.DataFrame({
    ...     'text': ['Hello world', 'Good morning', 'Hola mundo', 'Buenos días']
    ... })
    >>> y_train = pd.Series(['English', 'English', 'Spanish', 'Spanish'])
    >>> 
    >>> knn = CompressionKNN(k=1, data_column='text')
    >>> knn.fit(X_train, y_train)
    CompressionKNN(...)
    >>> 
    >>> X_test = pd.DataFrame({'text': ['Hi there', 'Hola amigo']})
    >>> knn.predict(X_test)
    0    English
    1    Spanish
    Name: target, dtype: object
    
    Using a custom compression function:
    
    >>> import zlib
    >>> 
    >>> def custom_compressor(data: bytes) -> bytes:
    ...     return zlib.compress(data, level=9)
    >>> 
    >>> knn = CompressionKNN(k=2, data_column='text', compressor=custom_compressor)
    >>> knn.fit(X_train, y_train)
    CompressionKNN(...)
    
    DNA sequence classification:
    
    >>> X_dna = pd.DataFrame({
    ...     'seq': ['ATCGATCG' * 10, 'ATCGATCG' * 10 + 'AT', 
    ...             'GCTAGCTA' * 8 + 'GGGG', 'TAGCTACG' * 8 + 'CCCC']
    ... })
    >>> y_dna = pd.Series(['Bacteria', 'Bacteria', 'Mammal', 'Mammal'])
    >>> 
    >>> knn = CompressionKNN(k=1, data_column='seq', compressor='lzma')
    >>> knn.fit(X_dna, y_dna)
    CompressionKNN(...)
    
    Notes
    -----
    The Normalized Compression Distance approximates the normalized information
    distance (NID), which is based on Kolmogorov complexity K(·):
    
    .. math::
        NID(x, y) = \\frac{\\max\\{K(x|y), K(y|x)\\}}{\\max\\{K(x), K(y)\\}}
    
    Since Kolmogorov complexity is uncomputable, we approximate K(x) with C(x),
    the compressed length using a real-world compressor. This makes NCD computable
    while preserving the theoretical properties of NID.
    
    **Advantages:**
    - No feature engineering or preprocessing required
    - Universal: works across different data modalities
    - Parameter-free distance metric (aside from k)
    - Theoretically grounded in information theory
    
    **Limitations:**
    - Computationally expensive (compression is slow)
    - Not suitable for large datasets
    - Performance depends on choosing appropriate compressor
    - Slower than traditional KNN with geometric distances
    
    References
    ----------
    .. [1] Cilibrasi, R., & Vitanyi, P. M. (2005). Clustering by compression.
           IEEE Transactions on Information theory, 51(4), 1523-1545.
    .. [2] Li, M., Chen, X., Li, X., Ma, B., & Vitányi, P. M. (2004). The 
           similarity metric. IEEE transactions on Information Theory, 50(12),
           3250-3264.
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
        """Fit the CompressionKNN classifier on training data.
        
        Stores the training data and precomputes compressed lengths for all
        training samples. Precomputation avoids redundant compression operations
        during prediction, improving efficiency when the same training data is
        used for multiple predictions.
        
        The method also validates input data, determines the data column if not
        specified, and prepares internal state for the prediction phase.
        
        Parameters
        ----------
        X_train : nw.DataFrame or nw.Series
            Training data containing samples to compress. Can be a DataFrame with
            one or more columns (specify data_column) or a Series.
        y_train : nw.DataFrame or nw.Series
            Target labels corresponding to X_train samples. Must have the same
            number of rows as X_train.
        
        Returns
        -------
        self : CompressionKNN
            Fitted classifier instance (enables method chaining).
        
        Examples
        --------
        >>> import pandas as pd
        >>> from explodingham.models.compression_learning.knn import CompressionKNN
        >>> 
        >>> X = pd.DataFrame({'text': ['cat', 'dog', 'mouse', 'rat']})
        >>> y = pd.Series(['mammal', 'mammal', 'mammal', 'mammal'])
        >>> 
        >>> knn = CompressionKNN(k=2, data_column='text')
        >>> knn.fit(X, y)
        CompressionKNN(...)
        
        Notes
        -----
        The fit method stores the entire training dataset in memory along with
        compressed lengths. For large datasets, memory usage can be significant.
        The compressed length C(x) is computed for each training sample x.
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
        """Predict class labels for test samples using compression-based similarity.
        
        Computes the Normalized Compression Distance (NCD) between each test sample
        and all training samples, identifies the k nearest neighbors, and predicts
        class labels via majority voting.
        
        The prediction process:
        1. Compress each test sample individually: C(y)
        2. Compress concatenations with training samples: C(xy)
        3. Compute NCD using precomputed C(x) from training data
        4. Find k nearest neighbors (smallest NCD values)
        5. Return mode (most frequent class) among k neighbors
        
        Parameters
        ----------
        X : nw.DataFrame or nw.Series
            Test samples to classify. Must have the same structure as training
            data (same column name if DataFrame).
        
        Returns
        -------
        nw.Series or nw.DataFrame
            Predicted class labels for each test sample. The return type matches
            the underlying dataframe backend (Pandas or Polars).
        
        Examples
        --------
        >>> import pandas as pd
        >>> from explodingham.models.compression_learning.knn import CompressionKNN
        >>> 
        >>> # Training
        >>> X_train = pd.DataFrame({'text': ['red apple', 'green apple', 
        ...                                    'red cherry', 'green grape']})
        >>> y_train = pd.Series(['fruit', 'fruit', 'fruit', 'fruit'])
        >>> knn = CompressionKNN(k=2, data_column='text').fit(X_train, y_train)
        >>> 
        >>> # Prediction
        >>> X_test = pd.DataFrame({'text': ['yellow banana', 'purple grape']})
        >>> predictions = knn.predict(X_test)
        >>> print(predictions)
        0    fruit
        1    fruit
        Name: target, dtype: object
        
        Notes
        -----
        The Normalized Compression Distance used is:
        
        .. math::
            NCD(x, y) = \\frac{C(xy) - \\min\\{C(x), C(y)\\}}{\\max\\{C(x), C(y)\\}}
        
        where:
        - C(x) = compressed length of x (precomputed during fit)
        - C(y) = compressed length of y (computed during predict)
        - C(xy) = compressed length of concatenation xy
        
        The denominator normalizes for different input lengths, making NCD
        scale-invariant. The numerator measures how much additional information
        y provides about x (or vice versa).
        
        Computational complexity is O(n·m·T) where n is test set size, m is
        training set size, and T is compression time per sample.
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
        """Compute compressed length for each value in a column.
        
        Creates a Narwhals expression that applies compression to each value
        in the specified column and returns the compressed byte length. This
        operation is lazy and will be executed as part of a larger dataframe
        computation pipeline.
        
        Parameters
        ----------
        column : str
            Name of the column containing data to compress. Values will be
            encoded to bytes (if strings) before compression.
        
        Returns
        -------
        nw.Expr
            Narwhals expression that maps each value to its compressed length
            as a 64-bit integer.
        
        Notes
        -----
        This method uses map_batches for efficiency, processing multiple values
        at once rather than one-by-one. The compressed lengths are returned as
        a NumPy array of int64 values for memory efficiency.
        """
        def compress_to_series(s):
            """Compress series values and return lengths as numpy array."""
            compressed_lengths = [len(self.compressor(x.encode(self.encoding) if isinstance(x, str) else x)) for x in s]
            return np.array(compressed_lengths, dtype=np.int64)
        
        return nw.col(column).map_batches(compress_to_series).cast(nw.Int64)
    
    def _handle_X(self, X: nw.DataFrame | nw.Series) -> nw.DataFrame:
        """Normalize input data to DataFrame format.
        
        Converts input to a Narwhals DataFrame, handling both Series and
        DataFrame inputs. If input is a Series, it's converted to a single-column
        DataFrame. This ensures consistent internal handling regardless of input type.
        
        Parameters
        ----------
        X : nw.DataFrame or nw.Series
            Input data to normalize. Can be a Pandas or Polars Series/DataFrame.
        
        Returns
        -------
        nw.DataFrame
            Input data as a Narwhals DataFrame (eager mode).
        """
        X = nw.from_native(X, allow_series=True, eager_only=True)
        
        if isinstance(X, nw.Series):
            X = X.to_frame()
        
        return X
        

    def _get_callable_compressor(self, compressor_name: str) -> callable:
        """Resolve compression algorithm name to callable function.
        
        Maps string identifiers to their corresponding compression functions
        from the Python standard library. This enables users to specify
        compressors by name rather than importing the functions themselves.
        
        Parameters
        ----------
        compressor_name : str
            Name of the compression algorithm. Must be one of:
            
            - 'gzip': General-purpose, fast (LZ77 + Huffman coding)
            - 'bz2': Better compression, slower (Burrows-Wheeler transform)
            - 'lzma': Best compression, slowest (Lempel-Ziv-Markov chain)
        
        Returns
        -------
        callable
            Compression function with signature `f(bytes) -> bytes`.
        
        Raises
        ------
        ValueError
            If compressor_name is not one of the supported options.
        
        Notes
        -----
        The choice of compressor affects both runtime and classification accuracy:
        
        - GZIP: Good default, balances speed and compression
        - BZ2: Better for repetitive patterns, slower than gzip
        - LZMA: Best for complex patterns (e.g., DNA), very slow
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
        """Add target labels to the internal model data.
        
        Horizontally concatenates target labels with feature data stored in
        model_data. Also handles target column naming: if y is a Series with
        no name, uses self.target_column; if y has a name, updates
        self.target_column to match.
        
        This method modifies self.model_data in place.
        
        Parameters
        ----------
        y : nw.DataFrame or nw.Series
            Target labels to add. If DataFrame, must have exactly one column.
            If Series, may have a name or be unnamed.
        
        Raises
        ------
        ValueError
            If y is a DataFrame with more than one column (multi-target not
            supported).
        
        Notes
        -----
        The method handles column naming automatically to ensure consistent
        target column identification across different input formats (named vs.
        unnamed Series, DataFrames with various column names).
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