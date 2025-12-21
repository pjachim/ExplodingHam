from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import re
import narwhals as nw
from typing import Any

# Maps python REGEX flags to their string representations in Rust ReGex single letter flags.
# Used for constructing regex patterns with inline flags.
# References:
# - Rust docs: https://docs.rs/regex/latest/regex/#grouping-and-flags
# - Python docs: https://docs.python.org/3/library/re.html#re.RegexFlag
REGEX_MAPPING = {
    re.IGNORECASE: 'i',
    re.MULTILINE: 'm',
    re.DOTALL: 's',
    re.VERBOSE: 'x',
    re.UNICODE: 'u'
}

class BinaryRegexClassifier(BaseExplodingHamClassifier):
    """
    Dataframe-aware regex classifier that matches patterns using configurable match types.
    
    This classifier uses the narwhals library to provide dataframe-agnostic pattern matching
    across multiple backends (pandas, polars, etc.). It supports both partial matching
    (pattern found anywhere in string) and full matching (entire string matches pattern).
    The classifier returns a dataframe column with configurable prediction values for
    matches and non-matches.
    
    Parameters
    ----------
    pattern : str, optional
        Regular expression pattern to match against (default is an empty string).
    flags : list[re.RegexFlag] | None, optional
        List of regex flags to modify pattern behavior (e.g., re.IGNORECASE, re.MULTILINE).
        These are converted to inline flags compatible with Rust regex syntax.
        Default is None.
    ignore_case : bool, optional
        If True, adds re.IGNORECASE flag to make pattern matching case-insensitive.
        This is a convenience parameter equivalent to including re.IGNORECASE in flags.
        Default is False.
    encoding : str, optional
        Character encoding for string operations (default is 'utf-8').
        Currently not actively used but reserved for future byte string handling.
    column_name : str | None, optional
        Name of the column to apply pattern matching to. If None, the input must be
        a Series or single-column DataFrame. Default is None.
    match_prediction : Any, optional
        Value to assign when pattern matches (default is 1).
    no_match_prediction : Any, optional
        Value to assign when pattern does not match (default is 0).
    match_type : str, optional
        Type of matching to perform. Valid options:
        - 'partial': Pattern can match anywhere in the string (uses str.contains)
        - 'full': Entire string must match the pattern (uses str.replace to check)
        Default is 'full'.
    prediction_name : str, optional
        Name of the output prediction column (default is 'prediction').
    
    Attributes
    ----------
    pattern : str
        The compiled pattern with inline flags (if any).
    flags : list[re.RegexFlag]
        List of regex flags being used.
    
    Examples
    --------
    >>> import pandas as pd
    >>> # Full match - entire string must match pattern
    >>> clf = BinaryRegexClassifier(pattern=r'\d{3}-\d{4}', match_type='full')
    >>> df = pd.DataFrame({'text': ['123-4567', 'call 123-4567', '999-0000']})
    >>> clf.predict(df)['prediction'].tolist()
    [1, 0, 1]
    
    >>> # Partial match - pattern found anywhere
    >>> clf = BinaryRegexClassifier(pattern=r'python', match_type='partial', ignore_case=True)
    >>> df = pd.DataFrame({'text': ['I love Python', 'Java is nice', 'PYTHON rocks']})
    >>> clf.predict(df)['prediction'].tolist()
    [1, 0, 1]
    
    >>> # Custom prediction values
    >>> clf = BinaryRegexClassifier(
    ...     pattern=r'error',
    ...     match_prediction='ERROR',
    ...     no_match_prediction='OK',
    ...     match_type='partial'
    ... )
    >>> df = pd.DataFrame({'logs': ['error in line 10', 'success', 'fatal error']})
    >>> clf.predict(df)['prediction'].tolist()
    ['ERROR', 'OK', 'ERROR']
    
    Notes
    -----
    This classifier uses narwhals for dataframe interoperability, allowing it to work
    seamlessly with pandas, polars, and other dataframe libraries. The regex pattern
    is executed using the underlying dataframe library's string operations.
    
    The 'full' match_type works by replacing all matches with empty string and checking
    if the result has length 0. This is equivalent to checking if the entire string
    matches the pattern.
    
    Python regex flags are automatically converted to inline Rust regex syntax using
    the REGEX_MAPPING dictionary, ensuring compatibility across different regex engines.
    """
    def __init__(
            self,
            pattern: str = '',
            flags: list[re.RegexFlag] | None = None,
            ignore_case: bool = False,
            encoding: str = 'utf-8',
            column_name: str | None = None,
            match_prediction: Any = 1,
            no_match_prediction: Any = 0,
            match_type: str = 'full',
            prediction_name: str = 'prediction'
        ) -> None:
        """
        Initialize the BinaryRegexClassifier with pattern and configuration.
        
        Parameters
        ----------
        pattern : str, optional
            Regular expression pattern to match against (default is an empty string).
        flags : list[re.RegexFlag] | None, optional
            List of regex flags to modify pattern behavior. Converted to inline flags.
            Supported flags: IGNORECASE, MULTILINE, DOTALL, VERBOSE, UNICODE.
            Default is None.
        ignore_case : bool, optional
            If True, adds re.IGNORECASE flag for case-insensitive matching.
            Default is False.
        encoding : str, optional
            Character encoding for string operations (default is 'utf-8').
        column_name : str | None, optional
            Name of column to match against. If None, input must be a Series or
            single-column DataFrame. Default is None.
        match_prediction : Any, optional
            Value returned when pattern matches (default is 1).
        no_match_prediction : Any, optional
            Value returned when pattern does not match (default is 0).
        match_type : str, optional
            Type of matching: 'partial' (anywhere in string) or 'full' (entire string).
            Default is 'full'.
        prediction_name : str, optional
            Name of the output prediction column (default is 'prediction').
        """
        # Process flags
        if (flags is not None) or ignore_case:
            # Dedupe flags
            flags_set = set(flags) if flags is not None else set()

            if ignore_case:
                flags_set.add(re.IGNORECASE)

            # Construct inline flag string
            flags_str = ''
            for flag in flags_set:
                flags_str += REGEX_MAPPING[flag]
            
            # Update pattern to include inline flags
            self._set_param('pattern', f'(?{flags_str})' + str(pattern), str)

            # Store flags as list for introspection
            self._set_param('flags', list(flags_set), list)

        else:
            flags_set = set()
            self._set_param('pattern', pattern, str)
            self._set_param('flags', [], list)
        
        self._set_param('encoding', encoding, str)
        self.column_name = column_name
        self._set_param('match_prediction', match_prediction, lambda x: x)
        self._set_param('no_match_prediction', no_match_prediction, lambda x: x)
        self._set_param('match_type', match_type, str)
        self._set_param('prediction_name', prediction_name, str)

    def fit(self, X: Any = None, y: Any = None) -> None:
        """
        Fit method for sklearn API compatibility. No training required for regex classifiers.
        
        Parameters
        ----------
        X : Any, optional
            Input data (not used). Default is None.
        y : Any, optional
            Target labels (not used). Default is None.
        """
        pass

    def predict(self, X: nw.DataFrame | nw.Series | Any) -> Any:
        """
        Apply regex pattern matching and return predictions as a dataframe.
        
        Parameters
        ----------
        X : DataFrame, Series, or compatible data structure
            Input data to classify. Can be a pandas/polars DataFrame, Series, or any
            structure supported by narwhals. If DataFrame with multiple columns,
            column_name must be specified in the constructor.
        
        Returns
        -------
        predictions : DataFrame
            Native dataframe (in the same format as input) containing the prediction
            column with match/no-match values.
        """
        X, column_name = self._process_dataframe(X)

        if self.match_type == 'partial':
            condition = nw.col(column_name).str.contains(self.pattern)
        elif self.match_type == 'full':
            condition = nw.col(column_name).str.replace(self.pattern, '').str.len_chars() == nw.lit(0)
        else:
            raise ValueError(f"Invalid match_type: {self.match_type}. Valid options are: ['partial', 'full']")  

        X = X.with_columns(
            nw.when(condition).then(nw.lit(self.match_prediction)).otherwise(nw.lit(self.no_match_prediction)).alias(self.prediction_name)
        )

        predictions = X.select(self.prediction_name).to_native()

        return predictions
    
    def _process_dataframe(self, X: nw.DataFrame | nw.Series) -> tuple[nw.DataFrame, str]:
        """
        Convert input to narwhals DataFrame and determine target column name.
        
        This helper method normalizes various input formats (DataFrame, Series, or
        native pandas/polars objects) into a narwhals DataFrame and identifies which
        column to apply pattern matching to.
        
        Parameters
        ----------
        X : DataFrame, Series, or compatible data structure
            Input data to process. Can be a narwhals DataFrame/Series or native
            pandas/polars objects that narwhals can convert.
        
        Returns
        -------
        df : nw.DataFrame
            Narwhals DataFrame ready for pattern matching operations.
        column_name : str
            Name of the column to apply regex matching to.
        
        Raises
        ------
        ValueError
            If input is a multi-column DataFrame and column_name was not specified
            in the constructor.
        
        Notes
        -----
        For Series input, the Series name is used as the column name. If the Series
        has no name, it is assigned 'input_column' as a default name.
        
        For DataFrame input, uses self.column_name if specified, otherwise requires
        exactly one column in the DataFrame.
        """
        X: nw.DataFrame = nw.from_native(X, allow_series=True)

        if type(X) is nw.Series:
            if X.name is None:
                X = X.alias('input_column')

            column_name = X.name
            X = X.to_frame()
        else:
            if self.column_name is not None:
                column_name = self.column_name
            else:
                if len(X.columns) > 1:
                    raise ValueError("If column name is not specified, input data must have exactly one column.")
                column_name = X.columns[0]

        return X, column_name