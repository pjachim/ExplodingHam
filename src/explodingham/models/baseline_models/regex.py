from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier
import re

class BaseRegexClassifier(BaseExplodingHamClassifier):
    def __init__(
            self,
            pattern: str = '',
            flags: list[re.RegexFlag] | None = None,
            ignore_case: bool = False,
            encoding: str = 'utf-8'
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
        
        # Combine flags for re.compile (bitwise OR)
        combined_flags = 0
        for flag in flags_set:
            combined_flags |= flag
        
        self.compiled_pattern = re.compile(self.pattern, combined_flags)

    def fit(self, X: list[str] | None = None, y: list | None = None) -> None:
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
        predictions = []
        for x in X:
            if isinstance(x, bytes):
                x = x.decode(self.encoding)
            match = self.compiled_pattern.search(x)
            predictions.append(match is not None)
        return predictions
    
class RegexPartialMatchClassifier(BaseRegexClassifier):
    """
    Classifier that matches a regex pattern only at the beginning of the input string.
    
    This classifier uses `re.match()` to check if the pattern matches at the start
    of the input text. It returns True only if the string begins with the pattern,
    False otherwise. This is useful for prefix matching, format validation, or
    enforcing that strings start with specific patterns (e.g., protocol identifiers,
    file extensions, or structured formats).
    
    Parameters
    ----------
    pattern : str, optional
        Regular expression pattern to match at the start of strings (default is an empty string).
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
    >>> # Validate URLs starting with http/https
    >>> clf = RegexPartialMatchClassifier(pattern=r'https?://')
    >>> urls = [
    ...     "https://example.com",
    ...     "ftp://files.com",
    ...     "http://site.org"
    ... ]
    >>> clf.predict(urls)
    [True, False, True]
    
    >>> # Check if strings start with digits
    >>> clf = RegexPartialMatchClassifier(pattern=r'\d+')
    >>> clf.predict(["123abc", "abc123", "456"])
    [True, False, True]
    
    >>> # Format validation: phone numbers starting with country code
    >>> clf = RegexPartialMatchClassifier(pattern=r'\+1-\d{3}-\d{3}-\d{4}')
    >>> clf.predict(["+1-555-123-4567", "555-123-4567", "+1-555-1234"])
    [True, False, False]
    
    Notes
    -----
    This classifier uses `re.match()` internally, which only checks for a match
    at the beginning of the string. To find patterns anywhere in the string,
    use `RegexFullMatchClassifier` with `re.search()` instead.
    
    If you want to ensure the entire string matches the pattern (not just the prefix),
    anchor your pattern with `^` at the start and `$` at the end.
    
    See Also
    --------
    RegexFullMatchClassifier : Matches pattern anywhere in the string
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
        predictions = []
        for x in X:
            if isinstance(x, bytes):
                x = x.decode(self.encoding)
            match = self.compiled_pattern.match(x)
            predictions.append(match is not None)
        return predictions