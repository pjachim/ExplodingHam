"""Unit tests for BinaryRegexClassifier."""

import pytest
import re
import pandas as pd
from explodingham.models.baseline_models.regex import BinaryRegexClassifier

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class TestBinaryRegexClassifierInitialization:
    """Test suite for BinaryRegexClassifier initialization."""
    
    def test_default_initialization(self) -> None:
        """Test BinaryRegexClassifier initializes with default parameters."""
        clf = BinaryRegexClassifier()
        assert clf.pattern == ''
        assert clf.flags == []
        assert clf.encoding == 'utf-8'
        assert clf.match_type == 'full'
        assert clf.match_prediction == 1
        assert clf.no_match_prediction == 0
        assert clf.prediction_name == 'prediction'
        assert clf.column_name is None
    
    def test_custom_pattern(self) -> None:
        """Test initialization with custom pattern."""
        clf = BinaryRegexClassifier(pattern=r'\d+')
        assert clf.pattern == r'\d+'
    
    def test_custom_encoding(self) -> None:
        """Test initialization with custom encoding."""
        clf = BinaryRegexClassifier(encoding='latin-1')
        assert clf.encoding == 'latin-1'
    
    def test_ignore_case_flag(self) -> None:
        """Test initialization with ignore_case parameter."""
        clf = BinaryRegexClassifier(pattern='test', ignore_case=True)
        assert re.IGNORECASE in clf.flags
        assert '(?i)' in clf.pattern
    
    def test_explicit_flags(self) -> None:
        """Test initialization with explicit flags list."""
        clf = BinaryRegexClassifier(pattern=r'^\d+', flags=[re.MULTILINE])
        assert re.MULTILINE in clf.flags
        assert '(?m)' in clf.pattern
    
    def test_flags_with_ignore_case(self) -> None:
        """Test that explicit flags and ignore_case are combined."""
        clf = BinaryRegexClassifier(
            pattern='test',
            flags=[re.MULTILINE],
            ignore_case=True
        )
        assert re.MULTILINE in clf.flags
        assert re.IGNORECASE in clf.flags
        # Both flags should be in inline format
        assert '(?' in clf.pattern
    
    def test_duplicate_flags_deduplicated(self) -> None:
        """Test that duplicate flags are deduplicated."""
        clf = BinaryRegexClassifier(
            pattern='test',
            flags=[re.IGNORECASE, re.IGNORECASE],
            ignore_case=True
        )
        # Should only have one IGNORECASE flag
        assert clf.flags.count(re.IGNORECASE) == 1
    
    def test_pattern_stored(self) -> None:
        """Test that pattern is stored during initialization."""
        clf = BinaryRegexClassifier(pattern=r'\d+')
        assert hasattr(clf, 'pattern')
        assert isinstance(clf.pattern, str)
    
    def test_empty_flags_list(self) -> None:
        """Test initialization with empty flags list."""
        clf = BinaryRegexClassifier(pattern='test', flags=[])
        assert clf.flags == []
    
    def test_custom_match_prediction(self) -> None:
        """Test initialization with custom match prediction value."""
        clf = BinaryRegexClassifier(match_prediction='MATCH')
        assert clf.match_prediction == 'MATCH'
    
    def test_custom_no_match_prediction(self) -> None:
        """Test initialization with custom no-match prediction value."""
        clf = BinaryRegexClassifier(no_match_prediction='NO_MATCH')
        assert clf.no_match_prediction == 'NO_MATCH'
    
    def test_match_type_partial(self) -> None:
        """Test initialization with partial match type."""
        clf = BinaryRegexClassifier(match_type='partial')
        assert clf.match_type == 'partial'
    
    def test_match_type_full(self) -> None:
        """Test initialization with full match type."""
        clf = BinaryRegexClassifier(match_type='full')
        assert clf.match_type == 'full'
    
    def test_custom_prediction_name(self) -> None:
        """Test initialization with custom prediction column name."""
        clf = BinaryRegexClassifier(prediction_name='result')
        assert clf.prediction_name == 'result'
    
    def test_custom_column_name(self) -> None:
        """Test initialization with custom column name."""
        clf = BinaryRegexClassifier(column_name='text_column')
        assert clf.column_name == 'text_column'


class TestBinaryRegexClassifierPartialMatch:
    """Test suite for BinaryRegexClassifier.predict with match_type='partial'."""
    
    def test_predict_match_anywhere(self) -> None:
        """Test that pattern matches anywhere in string with partial match."""
        clf = BinaryRegexClassifier(pattern=r'http', match_type='partial')
        df = pd.DataFrame({'text': ['http://example.com', 'https://site.org', 'ftp://files.com']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 0]
    
    def test_predict_substring_match(self) -> None:
        """Test that pattern matches as substring with partial match."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pd.DataFrame({'text': ['test case', 'this is a test', 'testing', 'no match']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 1, 0]
    
    def test_predict_with_digits(self) -> None:
        """Test matching digit patterns anywhere."""
        clf = BinaryRegexClassifier(pattern=r'\d+', match_type='partial')
        df = pd.DataFrame({'text': ['123abc', 'abc123', '456', 'nodigits']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 1, 0]
    
    def test_predict_url_validation(self) -> None:
        """Test URL validation with partial match."""
        clf = BinaryRegexClassifier(pattern=r'https?://', match_type='partial')
        df = pd.DataFrame({'text': [
            'https://example.com',
            'http://site.org',
            'ftp://files.com',
            'example.com'
        ]})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 0, 0]
    
    def test_predict_case_sensitive(self) -> None:
        """Test case-sensitive matching."""
        clf = BinaryRegexClassifier(pattern=r'Test', match_type='partial')
        df = pd.DataFrame({'text': ['Test case', 'test case', 'TEST case']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0, 0]
    
    def test_predict_case_insensitive(self) -> None:
        """Test case-insensitive matching with ignore_case."""
        clf = BinaryRegexClassifier(pattern=r'Test', ignore_case=True, match_type='partial')
        df = pd.DataFrame({'text': ['Test case', 'test case', 'TEST case']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 1]
    
    def test_predict_empty_string(self) -> None:
        """Test prediction with empty string."""
        clf = BinaryRegexClassifier(pattern=r'\w+', match_type='partial')
        df = pd.DataFrame({'text': ['', 'text']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [0, 1]
    
    def test_predict_empty_pattern(self) -> None:
        """Test prediction with empty pattern (matches everything)."""
        clf = BinaryRegexClassifier(pattern='', match_type='partial')
        df = pd.DataFrame({'text': ['text', '', '123']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 1]
    
    def test_predict_returns_dataframe(self) -> None:
        """Test that predict returns a dataframe."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pd.DataFrame({'text': ['test']})
        results = clf.predict(df)
        assert isinstance(results, pd.DataFrame)
    
    def test_predict_with_series(self) -> None:
        """Test prediction with Series input."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        series = pd.Series(['test case', 'no match'])
        results = clf.predict(series)['prediction'].tolist()
        assert results == [1, 0]
    
    def test_custom_prediction_values(self) -> None:
        """Test with custom match/no-match values."""
        clf = BinaryRegexClassifier(
            pattern=r'test',
            match_type='partial',
            match_prediction='FOUND',
            no_match_prediction='NOT_FOUND'
        )
        df = pd.DataFrame({'text': ['test', 'other']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == ['FOUND', 'NOT_FOUND']


class TestBinaryRegexClassifierFullMatch:
    """Test suite for BinaryRegexClassifier.predict with match_type='full'."""
    
    def test_predict_match_entire_string(self) -> None:
        """Test that entire string must match the pattern."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='full')
        df = pd.DataFrame({'text': ['test', 'test case', 'testing']})
        results = clf.predict(df)['prediction'].tolist()
        # Full match uses str.replace - only 'test' completely matches and becomes empty
        assert results == [1, 0, 0]
    
    def test_predict_email_pattern(self) -> None:
        """Test full email pattern matching."""
        clf = BinaryRegexClassifier(
            pattern=r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
            match_type='full'
        )
        df = pd.DataFrame({'text': [
            'user@example.com',
            'Contact user@example.com',
            'invalid.email'
        ]})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0, 0]
    
    def test_predict_keyword_detection(self) -> None:
        """Test keyword detection with full match."""
        clf = BinaryRegexClassifier(pattern=r'python', ignore_case=True, match_type='full')
        df = pd.DataFrame({'text': ['python', 'Python', 'PYTHON']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 1]
    
    def test_predict_phone_number(self) -> None:
        """Test phone number pattern."""
        clf = BinaryRegexClassifier(
            pattern=r'\d{3}-\d{4}',
            match_type='full'
        )
        df = pd.DataFrame({'text': ['123-4567', 'call 123-4567', '999-0000']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0, 1]
    
    def test_predict_digit_only(self) -> None:
        """Test matching strings that are entirely digits."""
        clf = BinaryRegexClassifier(pattern=r'\d+', match_type='full')
        df = pd.DataFrame({'text': ['12345', 'abc123', '67890']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0, 1]
    
    def test_predict_returns_dataframe(self) -> None:
        """Test that predict returns a dataframe."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='full')
        df = pd.DataFrame({'text': ['test']})
        results = clf.predict(df)
        assert isinstance(results, pd.DataFrame)
    
    def test_custom_prediction_column_name(self) -> None:
        """Test custom prediction column name."""
        clf = BinaryRegexClassifier(
            pattern=r'test',
            match_type='full',
            prediction_name='is_match'
        )
        df = pd.DataFrame({'text': ['test', 'other']})
        results = clf.predict(df)
        assert 'is_match' in results.columns
        assert results['is_match'].tolist() == [1, 0]


class TestBinaryRegexClassifierColumnHandling:
    """Test suite for column name handling and dataframe processing."""
    
    def test_single_column_dataframe_auto_select(self) -> None:
        """Test automatic column selection with single column."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pd.DataFrame({'my_column': ['test', 'other']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0]
    
    def test_multi_column_dataframe_requires_column_name(self) -> None:
        """Test that multi-column dataframe requires column_name parameter."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pd.DataFrame({
            'col1': ['test', 'other'],
            'col2': ['data', 'value']
        })
        with pytest.raises(ValueError, match="column name is not specified"):
            clf.predict(df)
    
    def test_multi_column_dataframe_with_column_name(self) -> None:
        """Test multi-column dataframe with specified column_name."""
        clf = BinaryRegexClassifier(
            pattern=r'test',
            match_type='partial',
            column_name='text_data'
        )
        df = pd.DataFrame({
            'text_data': ['test', 'other'],
            'metadata': ['info1', 'info2']
        })
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0]
    
    def test_series_with_name(self) -> None:
        """Test Series input with explicit name."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        series = pd.Series(['test', 'other'], name='my_series')
        results = clf.predict(series)
        assert 'prediction' in results.columns
        assert results['prediction'].tolist() == [1, 0]
    
    def test_series_without_name(self) -> None:
        """Test Series input without name (should assign default)."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        series = pd.Series(['test', 'other'])
        results = clf.predict(series)
        assert 'prediction' in results.columns
        assert results['prediction'].tolist() == [1, 0]


class TestBinaryRegexClassifierEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_dataframe(self) -> None:
        """Test prediction with empty dataframe."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        # Empty dataframe needs explicit dtype to work with .str accessor
        df = pd.DataFrame({'text': pd.Series([], dtype=str)})
        results = clf.predict(df)
        assert len(results) == 0
    
    def test_very_long_string(self) -> None:
        """Test with very long strings."""
        clf = BinaryRegexClassifier(pattern=r'needle', match_type='partial')
        long_text = 'a' * 10000 + 'needle' + 'b' * 10000
        df = pd.DataFrame({'text': [long_text]})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1]
    
    def test_special_regex_characters(self) -> None:
        """Test with special regex characters."""
        clf = BinaryRegexClassifier(pattern=r'\+1-\d{3}', match_type='partial')
        df = pd.DataFrame({'text': ['+1-555', '1-555']})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0]
    
    def test_complex_regex_pattern(self) -> None:
        """Test with complex regex pattern."""
        clf = BinaryRegexClassifier(
            pattern=r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
            match_type='partial'
        )
        df = pd.DataFrame({'text': [
            '1234 5678 9012 3456',
            '1234-5678-9012-3456',
            'just numbers 12345'
        ]})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 0]
    
    def test_invalid_match_type_raises_error(self) -> None:
        """Test that invalid match_type raises ValueError."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='invalid')
        df = pd.DataFrame({'text': ['test']})
        with pytest.raises(ValueError, match="Invalid match_type"):
            clf.predict(df)
    
    def test_whitespace_pattern(self) -> None:
        """Test matching whitespace patterns."""
        clf = BinaryRegexClassifier(pattern=r'\s+', match_type='partial')
        df = pd.DataFrame({'text': ['nospaces', 'has spaces', '\ttab']})
        results = clf.predict(df)['prediction'].tolist()
        # Only strings with whitespace should match
        assert results == [0, 1, 1]


class TestBinaryRegexClassifierIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_spam_detection_scenario(self) -> None:
        """Test spam detection use case."""
        clf = BinaryRegexClassifier(
            pattern=r'\b(free|win|winner|click|offer|prize|urgent|limited)\b',
            ignore_case=True,
            match_type='partial',
            match_prediction='SPAM',
            no_match_prediction='HAM'
        )
        df = pd.DataFrame({'message': [
            'You are the WINNER!',
            'Meeting tomorrow',
            'Click here for FREE offer',
            'Regular email',
            'URGENT: Limited time'
        ]})
        results = clf.predict(df)['prediction'].tolist()
        expected = ['SPAM', 'HAM', 'SPAM', 'HAM', 'SPAM']
        assert results == expected
    
    def test_url_validation_scenario(self) -> None:
        """Test URL validation use case."""
        clf = BinaryRegexClassifier(
            pattern=r'^https?://',
            match_type='partial',
            prediction_name='is_url'
        )
        df = pd.DataFrame({'link': [
            'https://example.com',
            'http://site.org',
            'ftp://files.com',
            'example.com',
            'https://secure.bank.com/login'
        ]})
        results = clf.predict(df)['is_url'].tolist()
        assert results == [1, 1, 0, 0, 1]
    
    def test_file_extension_validation(self) -> None:
        """Test file extension validation."""
        clf = BinaryRegexClassifier(
            pattern=r'\.py$',
            match_type='partial'
        )
        df = pd.DataFrame({'filename': [
            'main.py',
            'test_utils.py',
            'script.txt',
            'data.csv',
            'helper.py'
        ]})
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 1, 0, 0, 1]
    
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_dataframe_support(self) -> None:
        """Test that polars dataframes work via narwhals."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pl.DataFrame({'text': ['test', 'other']})
        results = clf.predict(df)
        # Results should be a polars DataFrame
        assert isinstance(results, pl.DataFrame)
        assert results['prediction'].to_list() == [1, 0]
    
    def test_fit_method_compatibility(self) -> None:
        """Test that fit method exists for sklearn compatibility."""
        clf = BinaryRegexClassifier(pattern=r'test', match_type='partial')
        df = pd.DataFrame({'text': ['test', 'other']})
        # Should not raise an error
        clf.fit(df)
        # Should still work after fit
        results = clf.predict(df)['prediction'].tolist()
        assert results == [1, 0]
