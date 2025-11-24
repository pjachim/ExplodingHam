"""Unit tests for Normalized Compression Distance."""

import zlib
import bz2
import pytest
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance


class TestNormalizedCompressionDistance:
    """Test suite for NormalizedCompressionDistance class."""
    
    @pytest.fixture
    def ncd_zlib(self) -> NormalizedCompressionDistance:
        """Create NCD instance with zlib compression."""
        return NormalizedCompressionDistance(zlib.compress)
    
    @pytest.fixture
    def ncd_bz2(self) -> NormalizedCompressionDistance:
        """Create NCD instance with bz2 compression."""
        return NormalizedCompressionDistance(bz2.compress)
    
    def test_initialization(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that NCD initializes correctly with a compressor."""
        assert ncd_zlib.compressor == zlib.compress
        assert callable(ncd_zlib.compressor)
    
    def test_compress_size_returns_positive(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that compress_size returns a positive integer."""
        size = ncd_zlib.compress_size(b"hello world")
        assert isinstance(size, int)
        assert size > 0
    
    def test_compress_size_with_empty_string(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test compress_size with empty input."""
        size = ncd_zlib.compress_size(b"")
        assert isinstance(size, int)
        assert size > 0  # Compressed empty string still has header
    
    def test_compress_size_compression_works(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that repeated data compresses to smaller size."""
        repeated_data = b"test" * 100
        size = ncd_zlib.compress_size(repeated_data)
        assert size < len(repeated_data)
    
    def test_ncd_identical_strings_returns_zero(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that NCD of identical strings is 0."""
        result = ncd_zlib.ncd(b"hello", b"hello")
        assert result == 0.0
    
    def test_ncd_returns_float(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that ncd returns a float value."""
        result = ncd_zlib.ncd(b"hello", b"world")
        assert isinstance(result, float)
    
    def test_ncd_range(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that NCD values are in expected range."""
        result = ncd_zlib.ncd(b"hello world", b"hello there")
        assert 0 <= result <= 1.5  # Upper bound can exceed 1 slightly
    
    def test_ncd_similar_strings_closer_than_different(
        self, ncd_zlib: NormalizedCompressionDistance
    ) -> None:
        """Test that similar strings have lower NCD than different strings."""
        similar = ncd_zlib.ncd(b"hello world", b"hello there")
        different = ncd_zlib.ncd(b"hello world", b"xyz abc 123")
        assert similar < different
    
    def test_ncd_symmetry(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test that NCD is approximately symmetric."""
        ncd_xy = ncd_zlib.ncd(b"hello", b"world")
        ncd_yx = ncd_zlib.ncd(b"world", b"hello")
        # Allow small floating point differences
        assert abs(ncd_xy - ncd_yx) < 0.01
    
    def test_ncd_with_bz2_compressor(self, ncd_bz2: NormalizedCompressionDistance) -> None:
        """Test NCD works with different compressor (bz2)."""
        result = ncd_bz2.ncd(b"test data", b"test info")
        assert isinstance(result, float)
        assert result >= 0
    
    def test_ncd_with_long_similar_strings(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD with longer strings that share common patterns."""
        doc1 = b"The quick brown fox jumps over the lazy dog" * 10
        doc2 = b"The quick brown fox jumps over the sleepy cat" * 10
        doc3 = b"Completely different content here" * 10
        
        ncd_similar = ncd_zlib.ncd(doc1, doc2)
        ncd_different = ncd_zlib.ncd(doc1, doc3)
        
        assert ncd_similar < ncd_different
    
    def test_ncd_empty_strings(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD with empty strings."""
        result = ncd_zlib.ncd(b"", b"")
        assert result == 0.0
    
    def test_ncd_one_empty_string(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD when one string is empty."""
        result = ncd_zlib.ncd(b"hello", b"")
        assert isinstance(result, float)
        assert result >= 0
    
    def test_ncd_repeated_patterns(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD with strings containing repeated patterns."""
        pattern1 = b"abc" * 50
        pattern2 = b"abc" * 50
        pattern3 = b"xyz" * 50
        
        assert ncd_zlib.ncd(pattern1, pattern2) == 0.0
        assert ncd_zlib.ncd(pattern1, pattern3) > 0
    
    def test_ncd_binary_data(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD with binary data."""
        data1 = bytes(range(256))
        data2 = bytes(range(256))
        data3 = bytes(range(128, 256)) + bytes(range(128))
        
        assert ncd_zlib.ncd(data1, data2) == 0.0
        assert ncd_zlib.ncd(data1, data3) > 0


class TestNormalizedCompressionDistanceEdgeCases:
    """Test edge cases and special scenarios for NCD."""
    
    @pytest.fixture
    def ncd_zlib(self) -> NormalizedCompressionDistance:
        """Create NCD instance with zlib compression."""
        return NormalizedCompressionDistance(zlib.compress)
    
    def test_ncd_with_unicode_like_bytes(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD with bytes representing unicode text."""
        text1 = "Hello 世界".encode('utf-8')
        text2 = "Hello World".encode('utf-8')
        
        result = ncd_zlib.ncd(text1, text2)
        assert isinstance(result, float)
        assert result > 0
    
    def test_ncd_single_character_difference(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD sensitivity to single character changes."""
        s1 = b"a" * 100
        s2 = b"a" * 99 + b"b"
        s3 = b"b" * 100
        
        ncd_small_diff = ncd_zlib.ncd(s1, s2)
        ncd_large_diff = ncd_zlib.ncd(s1, s3)
        
        assert ncd_small_diff < ncd_large_diff
    
    def test_ncd_substring_relationship(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD when one string is substring of another."""
        substring = b"hello"
        full_string = b"hello world how are you"
        unrelated = b"xyz abc def ghi jkl mno"
        
        ncd_substring = ncd_zlib.ncd(substring, full_string)
        ncd_unrelated = ncd_zlib.ncd(substring, unrelated)
        
        assert ncd_substring < ncd_unrelated
    
    def test_different_compressors_give_different_results(self) -> None:
        """Test that different compressors can yield different NCD values."""
        ncd_zlib = NormalizedCompressionDistance(zlib.compress)
        ncd_bz2 = NormalizedCompressionDistance(bz2.compress)
        
        data1 = b"test data here" * 20
        data2 = b"test info here" * 20
        
        result_zlib = ncd_zlib.ncd(data1, data2)
        result_bz2 = ncd_bz2.ncd(data1, data2)
        
        # Results may differ due to different compression algorithms
        assert isinstance(result_zlib, float)
        assert isinstance(result_bz2, float)
    
    def test_ncd_triangle_inequality_approximation(
        self, ncd_zlib: NormalizedCompressionDistance
    ) -> None:
        """Test that NCD roughly follows triangle inequality property."""
        x = b"abc" * 20
        y = b"abd" * 20
        z = b"xyz" * 20
        
        ncd_xy = ncd_zlib.ncd(x, y)
        ncd_yz = ncd_zlib.ncd(y, z)
        ncd_xz = ncd_zlib.ncd(x, z)
        
        # NCD is a metric approximation, so triangle inequality should roughly hold
        # d(x,z) <= d(x,y) + d(y,z) with some tolerance for compression artifacts
        assert ncd_xz <= ncd_xy + ncd_yz + 0.5  # Allow tolerance


class TestNormalizedCompressionDistanceIntegration:
    """Integration tests for real-world use cases."""
    
    @pytest.fixture
    def ncd_zlib(self) -> NormalizedCompressionDistance:
        """Create NCD instance with zlib compression."""
        return NormalizedCompressionDistance(zlib.compress)
    
    def test_document_similarity_use_case(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD for measuring document similarity."""
        doc1 = b"Machine learning is a subset of artificial intelligence"
        doc2 = b"Machine learning is part of AI technology"
        doc3 = b"Cooking recipes for Italian pasta dishes"
        
        similar_docs = ncd_zlib.ncd(doc1, doc2)
        different_docs = ncd_zlib.ncd(doc1, doc3)
        
        assert similar_docs < different_docs
        assert similar_docs < 0.7  # Similar docs should have low NCD
        assert different_docs > 0.3  # Different docs should have higher NCD
    
    def test_code_similarity_use_case(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD for detecting similar code snippets."""
        code1 = b"def hello():\n    print('hello')\n    return True"
        code2 = b"def hello():\n    print('world')\n    return True"
        code3 = b"class MyClass:\n    def __init__(self):\n        pass"
        
        similar_code = ncd_zlib.ncd(code1, code2)
        different_code = ncd_zlib.ncd(code1, code3)
        
        assert similar_code < different_code
    
    def test_plagiarism_detection_scenario(self, ncd_zlib: NormalizedCompressionDistance) -> None:
        """Test NCD for potential plagiarism detection."""
        original = b"The theory of relativity changed physics forever and Einstein proposed it"
        paraphrased = b"Einstein's relativity theory transformed physics permanently"
        unrelated = b"Quantum mechanics deals with subatomic particle behavior"
        
        ncd_paraphrase = ncd_zlib.ncd(original, paraphrased)
        ncd_unrelated = ncd_zlib.ncd(original, unrelated)
        
        # Paraphrased text should be somewhat similar
        assert ncd_paraphrase < ncd_unrelated
