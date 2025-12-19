"""Unit tests for Regex classifiers."""

import pytest
import re
from explodingham.models.baseline_models.regex import (
    BaseRegexClassifier,
    RegexPartialMatchClassifier,
    RegexFullMatchClassifier
)


class TestBaseRegexClassifierInitialization:
    """Test suite for BaseRegexClassifier initialization."""
    
    def test_default_initialization(self) -> None:
        """Test BaseRegexClassifier initializes with default parameters."""
        clf = BaseRegexClassifier()
        assert clf.pattern == ''
        assert clf.flags == []
        assert clf.encoding == 'utf-8'
        assert hasattr(clf, 'compiled_pattern')
    
    def test_custom_pattern(self) -> None:
        """Test initialization with custom pattern."""
        clf = BaseRegexClassifier(pattern=r'\d+')
        assert clf.pattern == r'\d+'
    
    def test_custom_encoding(self) -> None:
        """Test initialization with custom encoding."""
        clf = BaseRegexClassifier(encoding='latin-1')
        assert clf.encoding == 'latin-1'
    
    def test_ignore_case_flag(self) -> None:
        """Test initialization with ignore_case parameter."""
        clf = BaseRegexClassifier(pattern='test', ignore_case=True)
        assert re.IGNORECASE in clf.flags
    
    def test_explicit_flags(self) -> None:
        """Test initialization with explicit flags list."""
        clf = BaseRegexClassifier(pattern=r'^\d+', flags=[re.MULTILINE])
        assert re.MULTILINE in clf.flags
    
    def test_flags_with_ignore_case(self) -> None:
        """Test that explicit flags and ignore_case are combined."""
        clf = BaseRegexClassifier(
            pattern='test',
            flags=[re.MULTILINE],
            ignore_case=True
        )
        assert re.MULTILINE in clf.flags
        assert re.IGNORECASE in clf.flags
    
    def test_duplicate_flags_deduplicated(self) -> None:
        """Test that duplicate flags are deduplicated."""
        clf = BaseRegexClassifier(
            pattern='test',
            flags=[re.IGNORECASE, re.IGNORECASE],
            ignore_case=True
        )
        # Should only have one IGNORECASE flag
        assert clf.flags.count(re.IGNORECASE) == 1
    
    def test_compiled_pattern_created(self) -> None:
        """Test that compiled pattern is created during initialization."""
        clf = BaseRegexClassifier(pattern=r'\d+')
        assert hasattr(clf, 'compiled_pattern')
        assert isinstance(clf.compiled_pattern, re.Pattern)
    
    def test_empty_flags_list(self) -> None:
        """Test initialization with empty flags list."""
        clf = BaseRegexClassifier(pattern='test', flags=[])
        assert clf.flags == []


class TestRegexPartialMatchClassifierInitialization:
    """Test suite for RegexPartialMatchClassifier initialization."""
    
    def test_default_initialization(self) -> None:
        """Test RegexPartialMatchClassifier initializes with defaults."""
        clf = RegexPartialMatchClassifier()
        assert clf.pattern == ''
        assert clf.encoding == 'utf-8'
    
    def test_inherits_from_base(self) -> None:
        """Test that RegexPartialMatchClassifier inherits from BaseRegexClassifier."""
        clf = RegexPartialMatchClassifier()
        assert isinstance(clf, BaseRegexClassifier)


class TestRegexFullMatchClassifierInitialization:
    """Test suite for RegexFullMatchClassifier initialization."""
    
    def test_default_initialization(self) -> None:
        """Test RegexFullMatchClassifier initializes with defaults."""
        clf = RegexFullMatchClassifier()
        assert clf.pattern == ''
        assert clf.encoding == 'utf-8'
    
    def test_inherits_from_base(self) -> None:
        """Test that RegexFullMatchClassifier inherits from BaseRegexClassifier."""
        clf = RegexFullMatchClassifier()
        assert isinstance(clf, BaseRegexClassifier)


class TestRegexPartialMatchClassifierPredict:
    """Test suite for RegexPartialMatchClassifier.predict method."""
    
    def test_predict_match_at_start(self) -> None:
        """Test that pattern matches at the start of string."""
        clf = RegexPartialMatchClassifier(pattern=r'http')
        results = clf.predict(['http://example.com', 'https://site.org', 'ftp://files.com'])
        assert results == [True, True, False]
    
    def test_predict_no_match_in_middle(self) -> None:
        """Test that pattern in middle doesn't match."""
        clf = RegexPartialMatchClassifier(pattern=r'test')
        results = clf.predict(['test case', 'this is a test', 'testing'])
        assert results == [True, False, True]
    
    def test_predict_with_digits(self) -> None:
        """Test matching digit patterns at start."""
        clf = RegexPartialMatchClassifier(pattern=r'\d+')
        results = clf.predict(['123abc', 'abc123', '456'])
        assert results == [True, False, True]
    
    def test_predict_url_validation(self) -> None:
        """Test URL validation at start."""
        clf = RegexPartialMatchClassifier(pattern=r'https?://')
        results = clf.predict([
            'https://example.com',
            'http://site.org',
            'ftp://files.com',
            'example.com'
        ])
        assert results == [True, True, False, False]
    
    def test_predict_case_sensitive(self) -> None:
        """Test case-sensitive matching."""
        clf = RegexPartialMatchClassifier(pattern=r'Test')
        results = clf.predict(['Test case', 'test case', 'TEST case'])
        assert results == [True, False, False]
    
    def test_predict_case_insensitive(self) -> None:
        """Test case-insensitive matching with ignore_case."""
        clf = RegexPartialMatchClassifier(pattern=r'Test', ignore_case=True)
        results = clf.predict(['Test case', 'test case', 'TEST case'])
        assert results == [True, True, True]
    
    def test_predict_empty_string(self) -> None:
        """Test prediction with empty string."""
        clf = RegexPartialMatchClassifier(pattern=r'\w+')
        results = clf.predict(['', 'text'])
        assert results == [False, True]
    
    def test_predict_empty_pattern(self) -> None:
        """Test prediction with empty pattern (matches everything)."""
        clf = RegexPartialMatchClassifier(pattern='')
        results = clf.predict(['text', '', '123'])
        assert results == [True, True, True]
    
    def test_predict_returns_list(self) -> None:
        """Test that predict returns a list."""
        clf = RegexPartialMatchClassifier(pattern=r'test')
        results = clf.predict(['test'])
        assert isinstance(results, list)
    
    def test_predict_returns_booleans(self) -> None:
        """Test that predict returns boolean values."""
        clf = RegexPartialMatchClassifier(pattern=r'test')
        results = clf.predict(['test', 'no match'])
        assert all(isinstance(r, bool) for r in results)
    
    def test_predict_with_bytes(self) -> None:
        """Test prediction with byte strings."""
        clf = RegexPartialMatchClassifier(pattern=r'test', encoding='utf-8')
        results = clf.predict([b'test case', b'no match'])
        assert results == [True, False]
    
    def test_predict_mixed_string_and_bytes(self) -> None:
        """Test prediction with mixed string and bytes."""
        clf = RegexPartialMatchClassifier(pattern=r'test')
        results = clf.predict(['test', b'test', 'other'])
        assert results == [True, True, False]


class TestRegexFullMatchClassifierPredict:
    """Test suite for RegexFullMatchClassifier.predict method."""
    
    def test_predict_match_anywhere(self) -> None:
        """Test that pattern matches anywhere in string."""
        clf = RegexFullMatchClassifier(pattern=r'test')
        results = clf.predict(['test case', 'this is a test', 'testing', 'no match'])
        assert results == [True, True, True, False]
    
    def test_predict_email_detection(self) -> None:
        """Test email detection in text."""
        clf = RegexFullMatchClassifier(
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        results = clf.predict([
            'Contact support@example.com',
            'No email here',
            'Email: user@domain.org'
        ])
        assert results == [True, False, True]
    
    def test_predict_keyword_detection(self) -> None:
        """Test keyword detection anywhere."""
        clf = RegexFullMatchClassifier(pattern=r'python', ignore_case=True)
        results = clf.predict([
            'I love Python',
            'Java is nice',
            'PYTHON rocks'
        ])
        assert results == [True, False, True]
    
    def test_predict_multiple_patterns(self) -> None:
        """Test matching with alternation (multiple patterns)."""
        clf = RegexFullMatchClassifier(
            pattern=r'\b(free|win|prize)\b',
            ignore_case=True
        )
        results = clf.predict([
            'You WIN a prize!',
            'Get it for FREE',
            'Regular message'
        ])
        assert results == [True, True, False]
    
    def test_predict_hashtag_detection(self) -> None:
        """Test hashtag detection."""
        clf = RegexFullMatchClassifier(pattern=r'#[A-Za-z0-9_]+')
        results = clf.predict([
            'Love #Python',
            'Just text',
            '#AI and #ML'
        ])
        assert results == [True, False, True]
    
    def test_predict_social_handle(self) -> None:
        """Test social media handle detection."""
        clf = RegexFullMatchClassifier(pattern=r'@[A-Za-z0-9_]+')
        results = clf.predict([
            'Follow @user123',
            'Regular tweet',
            'Email at user@example.com'
        ])
        assert results == [True, False, True]
    
    def test_predict_multiline_flag(self) -> None:
        """Test using MULTILINE flag."""
        clf = RegexFullMatchClassifier(pattern=r'^\d+', flags=[re.MULTILINE])
        results = clf.predict([
            'text\n123 number',
            'no numbers\nat start',
            '456 start'
        ])
        assert results == [True, False, True]
    
    def test_predict_with_bytes(self) -> None:
        """Test prediction with byte strings."""
        clf = RegexFullMatchClassifier(pattern=r'test', encoding='utf-8')
        results = clf.predict([b'contains test here', b'no match'])
        assert results == [True, False]
    
    def test_predict_unicode(self) -> None:
        """Test prediction with Unicode strings."""
        clf = RegexFullMatchClassifier(pattern=r'café', encoding='utf-8')
        results = clf.predict(['I love café', 'Just coffee'])
        assert results == [True, False]
    
    def test_predict_returns_list(self) -> None:
        """Test that predict returns a list."""
        clf = RegexFullMatchClassifier(pattern=r'test')
        results = clf.predict(['test'])
        assert isinstance(results, list)


class TestRegexClassifierEncoding:
    """Test suite for encoding support in regex classifiers."""
    
    def test_utf8_encoding(self) -> None:
        """Test UTF-8 encoding with Unicode characters."""
        clf = RegexFullMatchClassifier(pattern=r'café', encoding='utf-8')
        results = clf.predict(['I love café'.encode('utf-8')])
        assert results == [True]
    
    def test_latin1_encoding(self) -> None:
        """Test Latin-1 encoding."""
        clf = RegexFullMatchClassifier(pattern=r'café', encoding='latin-1')
        text = 'café'.encode('latin-1')
        results = clf.predict([text])
        assert results == [True]
    
    def test_ascii_encoding(self) -> None:
        """Test ASCII encoding."""
        clf = RegexPartialMatchClassifier(pattern=r'hello', encoding='ascii')
        results = clf.predict([b'hello world'])
        assert results == [True]
    
    def test_mixed_string_bytes_encoding(self) -> None:
        """Test handling mixed string and bytes with encoding."""
        clf = RegexFullMatchClassifier(pattern=r'test', encoding='utf-8')
        results = clf.predict(['test string', b'test bytes'])
        assert results == [True, True]


class TestRegexClassifierEdgeCases:
    """Test edge cases and special scenarios for regex classifiers."""
    
    def test_empty_input_list(self) -> None:
        """Test prediction with empty input list."""
        clf = RegexPartialMatchClassifier(pattern=r'test')
        results = clf.predict([])
        assert results == []
    
    def test_very_long_string(self) -> None:
        """Test with very long strings."""
        clf = RegexFullMatchClassifier(pattern=r'needle')
        long_text = 'a' * 10000 + 'needle' + 'b' * 10000
        results = clf.predict([long_text])
        assert results == [True]
    
    def test_special_regex_characters(self) -> None:
        """Test with special regex characters."""
        clf = RegexPartialMatchClassifier(pattern=r'\+1-\d{3}')
        results = clf.predict(['+1-555', '1-555'])
        assert results == [True, False]
    
    def test_complex_regex_pattern(self) -> None:
        """Test with complex regex pattern."""
        clf = RegexFullMatchClassifier(
            pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        )
        results = clf.predict([
            '1234 5678 9012 3456',
            '1234-5678-9012-3456',
            'just numbers 12345'
        ])
        assert results == [True, True, False]
    
    def test_no_match_anywhere(self) -> None:
        """Test when pattern doesn't match anywhere."""
        clf = RegexFullMatchClassifier(pattern=r'xyz123')
        results = clf.predict(['abc', 'def', '123'])
        assert results == [False, False, False]
    
    def test_all_match(self) -> None:
        """Test when pattern matches all inputs."""
        clf = RegexPartialMatchClassifier(pattern=r'\w+')
        results = clf.predict(['test', 'abc', '123'])
        assert results == [True, True, True]
    
    def test_whitespace_pattern(self) -> None:
        """Test matching whitespace patterns."""
        clf = RegexFullMatchClassifier(pattern=r'\s+')
        results = clf.predict(['no spaces', 'has spaces', '\ttab'])
        assert results == [True, True, True]
    
    def test_anchor_patterns(self) -> None:
        """Test with anchor patterns."""
        clf = RegexPartialMatchClassifier(pattern=r'^test$')
        results = clf.predict(['test', 'test case', 'testing'])
        assert results == [True, False, False]


class TestRegexClassifierIntegration:
    """Integration tests for regex classifiers with realistic scenarios."""
    
    def test_url_validation_scenario(self) -> None:
        """Test URL validation use case."""
        clf = RegexPartialMatchClassifier(pattern=r'https?://')
        urls = [
            'https://example.com',
            'http://site.org',
            'ftp://files.com',
            'example.com',
            'https://secure.bank.com/login'
        ]
        results = clf.predict(urls)
        assert results == [True, True, False, False, True]
    
    def test_spam_detection_scenario(self) -> None:
        """Test spam detection use case."""
        clf = RegexFullMatchClassifier(
            pattern=r'\b(free|win|winner|click|offer|prize|urgent|limited)\b',
            ignore_case=True
        )
        messages = [
            'You are the WINNER!',
            'Meeting tomorrow',
            'Click here for FREE offer',
            'Regular email',
            'URGENT: Limited time'
        ]
        results = clf.predict(messages)
        expected_spam = [True, False, True, False, True]
        assert results == expected_spam
    
    def test_phone_number_validation_scenario(self) -> None:
        """Test phone number validation use case."""
        clf = RegexPartialMatchClassifier(pattern=r'\+1-\d{3}-\d{3}-\d{4}')
        phone_numbers = [
            '+1-555-123-4567',
            '555-123-4567',
            '+1-800-555-0199',
            '1-555-123-4567',
            '+1-555-1234'
        ]
        results = clf.predict(phone_numbers)
        assert results == [True, False, True, False, False]
    
    def test_file_extension_validation_scenario(self) -> None:
        """Test file extension validation use case."""
        clf = RegexPartialMatchClassifier(pattern=r'[a-zA-Z0-9_]+\.py$')
        filenames = [
            'main.py',
            'test_utils.py',
            'script.txt',
            'data.csv',
            'helper_functions.py'
        ]
        results = clf.predict(filenames)
        assert results == [True, True, False, False, True]
    
    def test_code_comment_detection_scenario(self) -> None:
        """Test code comment detection use case."""
        clf = RegexPartialMatchClassifier(pattern=r'\s*#')
        code_lines = [
            '# This is a comment',
            'x = 42  # Inline comment',
            '    # Indented comment',
            "print('Hello')",
            '#TODO: Fix this'
        ]
        results = clf.predict(code_lines)
        # Lines 0, 2, 4 start with # (with optional whitespace)
        assert results == [True, False, True, False, True]
    
    def test_date_format_validation_scenario(self) -> None:
        """Test date format validation use case."""
        clf = RegexPartialMatchClassifier(pattern=r'\d{4}-\d{2}-\d{2}')
        log_entries = [
            '2025-12-18 System started',
            'Error occurred yesterday',
            '2024-01-01 New year log',
            'Status: OK',
            '2025-06-15 Maintenance completed'
        ]
        results = clf.predict(log_entries)
        assert results == [True, False, True, False, True]
    
    def test_comparison_partial_vs_full(self) -> None:
        """Test comparing partial vs full match behavior."""
        pattern = r'test'
        
        partial_clf = RegexPartialMatchClassifier(pattern=pattern)
        full_clf = RegexFullMatchClassifier(pattern=pattern)
        
        test_strings = [
            'test case',
            'this is a test',
            'testing',
            'no match',
            'retest'
        ]
        
        partial_results = partial_clf.predict(test_strings)
        full_results = full_clf.predict(test_strings)
        
        # Partial matches: 0, 2 (start with 'test')
        assert partial_results == [True, False, True, False, False]
        
        # Full matches: 0, 1, 2, 4 (contain 'test')
        assert full_results == [True, True, True, False, True]
