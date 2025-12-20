from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import re
import narwhals as nw
from typing import Any

class BaseRegexClassifier(BaseExplodingHamClassifier):
    def __init__(
            self,
            pattern: str = '',
            flags: list[re.RegexFlag] | None = None,
            ignore_case: bool = False,
            encoding: str = 'utf-8',
            column_name: str | None = None
        ) -> None:
        """
        Regex-based classifier that uses a regular expression pattern to classify input data.
        Parameters
        ----------
        pattern : str, optional
            Regular expression pattern to use for classification (default is an empty string).
        flags : list[re.RegexFlag] | None, optional
            Flags to pass to the regex compiler (default is None).
        encoding : str, optional
            Encoding to use when converting strings to bytes (default is 'utf-8').
        """
        self._set_param('pattern', pattern, str)

        # Dedupe flags
        flags_set = set(flags) if flags is not None else set()

        # Add ignore case flag if specified, and not already present
        if ignore_case:
            flags_set.add(re.IGNORECASE)

        # Store flags as list for introspection
        self._set_param('flags', list(flags_set), list)
        self._set_param('encoding', encoding, str)
        self._set_param('column_name', column_name, (str, type(None)))
        
        # Combine flags for re.compile (bitwise OR)
        combined_flags = 0
        for flag in flags_set:
            combined_flags |= flag
        
        self.compiled_pattern = re.compile(self.pattern, combined_flags)

    def fit(self, X: Any = None, y: Any = None) -> None:
        """
        Fit method for compatibility. No training is required for regex classifiers.
        Parameters
        ----------
        X : list[str] | None, optional
            Input data (not used).
        y : list | None, optional
            Target labels (not used).
        """
        pass

class RegexFullMatchClassifier(BaseRegexClassifier):
    """
    Classifier that detects partial matches of a regex pattern anywhere in the input string.
    
    This classifier uses `re.search()` to find the pattern anywhere within the input text.
    It returns True if the pattern appears at any position in the string, False otherwise.
    This is useful for keyword detection, substring matching, or identifying documents
    that contain specific patterns regardless of their position.
    
    Parameters
    ----------
    pattern : str, optional
        Regular expression pattern to search for (default is an empty string).
    flags : list[re.RegexFlag] | None, optional
        List of regex flags to modify pattern behavior (e.g., re.IGNORECASE, re.MULTILINE).
        Default is None.
    ignore_case : bool, optional
        If True, adds re.IGNORECASE flag to make pattern matching case-insensitive.
        Default is False.
    encoding : str, optional
        Character encoding to use when decoding byte strings (default is 'utf-8').
    
    Attributes
    ----------
    compiled_pattern : re.Pattern
        Compiled regular expression pattern ready for matching operations.
    
    Examples
    --------
    >>> # Detect emails anywhere in text
    >>> clf = RegexFullMatchClassifier(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    >>> texts = [
    ...     "Contact us at support@example.com for help",
    ...     "No contact information here",
    ...     "My email is user@domain.org"
    ... ]
    >>> clf.predict(texts)
    [True, False, True]
    
    >>> # Case-insensitive keyword detection
    >>> clf = RegexFullMatchClassifier(pattern=r'python', ignore_case=True)
    >>> clf.predict(["I love Python", "Java is nice", "PYTHON rocks"])
    [True, False, True]
    
    Notes
    -----
    This classifier uses `re.search()` internally, which scans through the string
    looking for the first location where the pattern matches. For full string matching
    (anchored at start), use `RegexPartialMatchClassifier` instead.
    
    See Also
    --------
    RegexPartialMatchClassifier : Matches pattern only at the start of the string
    """
    def predict(self, X: list[str]) -> list[bool]:
        """
        Predict whether each input string matches the regex pattern.
        Parameters
        ----------
        X : list[str]
            List of input strings to classify.
        Returns
        -------
        predictions : list[bool]
            List of boolean values indicating whether each input matches the pattern.
        """
        X, column_name = self._process_dataframe(X)

        X.with_columns(
            nw.col(column_name).str.replace()
        )

        predictions = []
        for x in X:
            if isinstance(x, bytes):
                x = x.decode(self.encoding)
            match = self.compiled_pattern.search(x)
            predictions.append(match is not None)
        
        return predictions
    
    def _process_dataframe(self, X: Any) -> tuple[nw.DataFrame, str]:
        X: nw.DataFrame = nw.from_native(X)

        if type(X) is not nw.DataFrame:
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