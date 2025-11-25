"""Unit tests for CompressionKNN classifier."""

import pytest
import numpy as np
from explodingham.models.compression_learning.knn import CompressionKNN
from explodingham.utils.distance_metrics.ncd import NormalizedCompressionDistance


class TestCompressionKNNInitialization:
    """Test suite for CompressionKNN initialization."""
    
    def test_default_initialization(self) -> None:
        """Test CompressionKNN initializes with default parameters."""
        clf = CompressionKNN()
        assert clf.n_neighbors == 5
        assert clf.compression == 'gzip'
        assert clf.token_scrubbing is False
        assert clf.encoding == 'utf-8'
        assert isinstance(clf.ncd, NormalizedCompressionDistance)
    
    def test_custom_n_neighbors(self) -> None:
        """Test initialization with custom n_neighbors."""
        clf = CompressionKNN(n_neighbors=3)
        assert clf.n_neighbors == 3
    
    def test_custom_n_neighbors_large(self) -> None:
        """Test initialization with large n_neighbors."""
        clf = CompressionKNN(n_neighbors=100)
        assert clf.n_neighbors == 100
    
    def test_custom_compression_gzip(self) -> None:
        """Test initialization with gzip compression."""
        clf = CompressionKNN(compression='gzip')
        assert clf.compression == 'gzip'
    
    def test_custom_token_scrubbing_true(self) -> None:
        """Test initialization with token_scrubbing enabled."""
        clf = CompressionKNN(token_scrubbing=True)
        assert clf.token_scrubbing is True
    
    def test_custom_token_scrubbing_false(self) -> None:
        """Test initialization with token_scrubbing disabled."""
        clf = CompressionKNN(token_scrubbing=False)
        assert clf.token_scrubbing is False
    
    def test_custom_encoding_utf8(self) -> None:
        """Test initialization with UTF-8 encoding."""
        clf = CompressionKNN(encoding='utf-8')
        assert clf.encoding == 'utf-8'
    
    def test_custom_encoding_latin1(self) -> None:
        """Test initialization with Latin-1 encoding."""
        clf = CompressionKNN(encoding='latin-1')
        assert clf.encoding == 'latin-1'
    
    def test_custom_encoding_ascii(self) -> None:
        """Test initialization with ASCII encoding."""
        clf = CompressionKNN(encoding='ascii')
        assert clf.encoding == 'ascii'
    
    def test_all_custom_parameters(self) -> None:
        """Test initialization with all custom parameters."""
        clf = CompressionKNN(n_neighbors=7, compression='gzip', token_scrubbing=True, encoding='latin-1')
        assert clf.n_neighbors == 7
        assert clf.compression == 'gzip'
        assert clf.token_scrubbing is True
        assert clf.encoding == 'latin-1'
    
    def test_ncd_instance_created(self) -> None:
        """Test that NCD instance is created during initialization."""
        clf = CompressionKNN()
        assert hasattr(clf, 'ncd')
        assert isinstance(clf.ncd, NormalizedCompressionDistance)
        assert callable(clf.ncd.ncd)
    
    def test_invalid_compression_method(self) -> None:
        """Test initialization fails with invalid compression method."""
        with pytest.raises(ValueError) as exc_info:
            CompressionKNN(compression='invalid')
        assert 'Invalid option: invalid' in str(exc_info.value)
    
    def test_invalid_n_neighbors_type(self) -> None:
        """Test initialization fails with invalid n_neighbors type."""
        with pytest.raises(ValueError) as exc_info:
            CompressionKNN(n_neighbors='five')
        assert 'Invalid value for n_neighbors' in str(exc_info.value)
    
    def test_invalid_token_scrubbing_type(self) -> None:
        """Test initialization with string for token_scrubbing."""
        # String 'yes' will be converted to True by bool(), so this should work
        clf = CompressionKNN(token_scrubbing='yes')
        assert clf.token_scrubbing is True


class TestCompressionKNNGetCompressionFunction:
    """Test suite for CompressionKNN._get_compression_function method."""
    
    def test_get_compression_function_gzip(self) -> None:
        """Test _get_compression_function returns gzip.compress."""
        clf = CompressionKNN(compression='gzip')
        compressor = clf._get_compression_function()
        assert callable(compressor)
        
        # Test that it actually compresses
        import gzip
        test_data = b"test data"
        compressed = compressor(test_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    
    def test_get_compression_function_is_gzip_compress(self) -> None:
        """Test that returned function is gzip.compress."""
        import gzip
        clf = CompressionKNN(compression='gzip')
        compressor = clf._get_compression_function()
        assert compressor == gzip.compress
    
    def test_compression_function_works(self) -> None:
        """Test that compression function actually compresses data."""
        clf = CompressionKNN()
        compressor = clf._get_compression_function()
        
        data = b"a" * 1000
        compressed = compressor(data)
        assert len(compressed) < len(data)
    
    def test_unsupported_compression_method(self) -> None:
        """Test _get_compression_function raises NotImplementedError for unsupported methods."""
        # Manually set compression to bypass validation
        clf = CompressionKNN()
        clf.compression = 'bz2'
        
        with pytest.raises(NotImplementedError) as exc_info:
            clf._get_compression_function()
        assert "Compression method 'bz2' is not implemented" in str(exc_info.value)


class TestCompressionKNNConvertToBytes:
    """Test suite for CompressionKNN._convert_to_bytes method."""
    
    def test_convert_string_to_bytes_utf8(self) -> None:
        """Test converting string to bytes with UTF-8 encoding."""
        clf = CompressionKNN(encoding='utf-8')
        result = clf._convert_to_bytes('hello')
        assert result == b'hello'
        assert isinstance(result, bytes)
    
    def test_convert_string_to_bytes_latin1(self) -> None:
        """Test converting string to bytes with Latin-1 encoding."""
        clf = CompressionKNN(encoding='latin-1')
        result = clf._convert_to_bytes('café')
        assert isinstance(result, bytes)
        assert result == 'café'.encode('latin-1')
    
    def test_convert_string_to_bytes_ascii(self) -> None:
        """Test converting string to bytes with ASCII encoding."""
        clf = CompressionKNN(encoding='ascii')
        result = clf._convert_to_bytes('hello')
        assert result == b'hello'
    
    def test_convert_bytes_remains_bytes(self) -> None:
        """Test that bytes input remains unchanged."""
        clf = CompressionKNN()
        input_bytes = b'hello world'
        result = clf._convert_to_bytes(input_bytes)
        assert result == input_bytes
        assert result is input_bytes
    
    def test_convert_unicode_string(self) -> None:
        """Test converting Unicode string to bytes."""
        clf = CompressionKNN(encoding='utf-8')
        result = clf._convert_to_bytes('héllo wörld 🌍')
        assert isinstance(result, bytes)
        assert result == 'héllo wörld 🌍'.encode('utf-8')
    
    def test_convert_empty_string(self) -> None:
        """Test converting empty string to bytes."""
        clf = CompressionKNN()
        result = clf._convert_to_bytes('')
        assert result == b''
    
    def test_convert_empty_bytes(self) -> None:
        """Test that empty bytes remain unchanged."""
        clf = CompressionKNN()
        result = clf._convert_to_bytes(b'')
        assert result == b''
    
    def test_convert_with_different_encodings(self) -> None:
        """Test that encoding parameter affects conversion."""
        text = 'café'
        
        clf_utf8 = CompressionKNN(encoding='utf-8')
        result_utf8 = clf_utf8._convert_to_bytes(text)
        
        clf_latin1 = CompressionKNN(encoding='latin-1')
        result_latin1 = clf_latin1._convert_to_bytes(text)
        
        # Different encodings produce different byte sequences
        assert result_utf8 != result_latin1


class TestCompressionKNNFit:
    """Test suite for CompressionKNN.fit method."""
    
    @pytest.fixture
    def simple_data(self) -> tuple:
        """Create simple training data."""
        X = [b"hello world", b"hello there", b"goodbye world", b"goodbye there"]
        y = [0, 0, 1, 1]
        return X, y
    
    def test_fit_stores_training_data(self, simple_data: tuple) -> None:
        """Test that fit stores training data."""
        X, y = simple_data
        clf = CompressionKNN()
        result = clf.fit(X, y)
        
        assert hasattr(clf, 'stored_X')
        assert hasattr(clf, 'stored_y')
        # Data is converted to bytes, so check as bytes
        assert all(isinstance(x, bytes) for x in clf.stored_X)
        assert clf.stored_y == y
    
    def test_fit_returns_self(self, simple_data: tuple) -> None:
        """Test that fit returns self for chaining."""
        X, y = simple_data
        clf = CompressionKNN()
        result = clf.fit(X, y)
        
        assert result is clf
    
    def test_fit_with_numpy_array_labels(self) -> None:
        """Test fit with numpy array labels."""
        X = [b"data1", b"data2", b"data3"]
        y = np.array([0, 1, 0])
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert hasattr(clf, 'stored_y')
        np.testing.assert_array_equal(clf.stored_y, y)
    
    def test_fit_with_list_labels(self) -> None:
        """Test fit with list labels."""
        X = [b"data1", b"data2", b"data3"]
        y = [0, 1, 0]
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert clf.stored_y == y
    
    def test_fit_with_string_labels(self) -> None:
        """Test fit with string labels."""
        X = [b"cat behavior", b"dog behavior", b"cat play"]
        y = ['cat', 'dog', 'cat']
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert clf.stored_y == y
    
    def test_fit_with_single_sample(self) -> None:
        """Test fit with single training sample."""
        X = [b"single sample"]
        y = [1]
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert clf.stored_X == X
        assert clf.stored_y == y
    
    def test_fit_with_many_samples(self) -> None:
        """Test fit with many training samples."""
        X = [b"sample %d" % i for i in range(100)]
        y = [i % 3 for i in range(100)]
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert len(clf.stored_X) == 100
        assert len(clf.stored_y) == 100
    
    def test_fit_overwrites_previous_data(self, simple_data: tuple) -> None:
        """Test that calling fit again overwrites previous training data."""
        X1, y1 = simple_data
        X2 = [b"new data"]
        y2 = [99]
        
        clf = CompressionKNN()
        clf.fit(X1, y1)
        clf.fit(X2, y2)
        
        assert len(clf.stored_X) == 1
        assert clf.stored_y == y2
    
    def test_fit_with_string_inputs(self) -> None:
        """Test fit automatically converts string inputs to bytes."""
        X = ["hello world", "hello there", "goodbye world"]
        y = [0, 0, 1]
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert hasattr(clf, 'stored_X')
        assert all(isinstance(x, bytes) for x in clf.stored_X)
        assert clf.stored_X[0] == b"hello world"
    
    def test_fit_with_mixed_string_and_bytes(self) -> None:
        """Test fit handles mixed string and bytes inputs."""
        X = ["hello", b"world", "test"]
        y = [0, 1, 0]
        
        clf = CompressionKNN()
        clf.fit(X, y)
        
        assert all(isinstance(x, bytes) for x in clf.stored_X)
        assert clf.stored_X[0] == b"hello"
        assert clf.stored_X[1] == b"world"
    
    def test_fit_with_unicode_strings(self) -> None:
        """Test fit handles Unicode strings correctly."""
        X = ["café", "naïve", "résumé"]
        y = [0, 1, 0]
        
        clf = CompressionKNN(encoding='utf-8')
        clf.fit(X, y)
        
        assert all(isinstance(x, bytes) for x in clf.stored_X)
        assert clf.stored_X[0] == "café".encode('utf-8')
    
    def test_fit_with_different_encoding(self) -> None:
        """Test fit respects encoding parameter."""
        X = ["café", "naïve"]
        y = [0, 1]
        
        clf_utf8 = CompressionKNN(encoding='utf-8')
        clf_utf8.fit(X, y)
        
        clf_latin1 = CompressionKNN(encoding='latin-1')
        clf_latin1.fit(X, y)
        
        # Different encodings produce different byte sequences
        assert clf_utf8.stored_X[0] != clf_latin1.stored_X[0]


class TestCompressionKNNPredict:
    """Test suite for CompressionKNN.predict method."""
    
    @pytest.fixture
    def trained_classifier(self) -> CompressionKNN:
        """Create a trained classifier with simple data."""
        X = [
            b"hello world hello world",
            b"hello there hello there",
            b"goodbye world goodbye world",
            b"goodbye there goodbye there"
        ]
        y = [0, 0, 1, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X, y)
        return clf
    
    def test_predict_single_sample(self, trained_classifier: CompressionKNN) -> None:
        """Test predict with single sample."""
        X_test = [b"hello world"]
        predictions = trained_classifier.predict(X_test)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    def test_predict_multiple_samples(self, trained_classifier: CompressionKNN) -> None:
        """Test predict with multiple samples."""
        X_test = [b"hello world", b"goodbye world"]
        predictions = trained_classifier.predict(X_test)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 2
    
    def test_predict_similar_to_class_0(self, trained_classifier: CompressionKNN) -> None:
        """Test predict returns class 0 for samples similar to class 0."""
        X_test = [b"hello world hello hello"]
        predictions = trained_classifier.predict(X_test)
        
        assert predictions[0] == 0
    
    def test_predict_similar_to_class_1(self, trained_classifier: CompressionKNN) -> None:
        """Test predict returns class 1 for samples similar to class 1."""
        X_test = [b"goodbye world goodbye goodbye"]
        predictions = trained_classifier.predict(X_test)
        
        assert predictions[0] == 1
    
    def test_predict_returns_list(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict returns a list."""
        X_test = [b"test"]
        predictions = trained_classifier.predict(X_test)
        
        assert isinstance(predictions, list)
    
    def test_predict_with_k_1(self) -> None:
        """Test predict with k=1 (nearest neighbor)."""
        X_train = [b"aaa", b"bbb"]
        y_train = [0, 1]
        
        clf = CompressionKNN(n_neighbors=1)
        clf.fit(X_train, y_train)
        
        # Sample closer to "aaa"
        predictions = clf.predict([b"aaa"])
        assert predictions[0] == 0
        
        # Sample closer to "bbb"
        predictions = clf.predict([b"bbb"])
        assert predictions[0] == 1
    
    def test_predict_with_string_labels(self) -> None:
        """Test predict with string class labels."""
        X_train = [b"cat meow", b"dog bark", b"cat purr"]
        y_train = ['cat', 'dog', 'cat']
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"cat meow purr"])
        assert predictions[0] == 'cat'
    
    def test_predict_consistency(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict returns consistent results for same input."""
        X_test = [b"hello world"]
        
        predictions1 = trained_classifier.predict(X_test)
        predictions2 = trained_classifier.predict(X_test)
        
        assert predictions1 == predictions2
    
    def test_predict_empty_input(self, trained_classifier: CompressionKNN) -> None:
        """Test predict with empty input list."""
        X_test = []
        predictions = trained_classifier.predict(X_test)
        
        assert predictions == []
    
    def test_predict_with_string_inputs(self) -> None:
        """Test predict automatically converts string inputs to bytes."""
        X_train = ["hello world", "hello there", "goodbye world", "goodbye there"]
        y_train = [0, 0, 1, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        # Test with string inputs
        X_test = ["hello", "goodbye"]
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 2
        assert predictions[0] in [0, 1]
        assert predictions[1] in [0, 1]
    
    def test_predict_with_mixed_string_and_bytes(self) -> None:
        """Test predict handles mixed string and bytes inputs."""
        X_train = ["hello", "goodbye"]
        y_train = [0, 1]
        
        clf = CompressionKNN(n_neighbors=1)
        clf.fit(X_train, y_train)
        
        # Mix of string and bytes
        X_test = ["hello", b"goodbye"]
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 2
    
    def test_predict_with_unicode_strings(self) -> None:
        """Test predict handles Unicode strings correctly."""
        X_train = ["café", "naïve", "résumé"]
        y_train = [0, 1, 0]
        
        clf = CompressionKNN(n_neighbors=1, encoding='utf-8')
        clf.fit(X_train, y_train)
        
        X_test = ["café"]
        predictions = clf.predict(X_test)
        
        assert predictions[0] == 0


class TestCompressionKNNPredictProba:
    """Test suite for CompressionKNN.predict_proba method."""
    
    @pytest.fixture
    def trained_classifier(self) -> CompressionKNN:
        """Create a trained classifier with simple data."""
        X = [
            b"hello world",
            b"hello there",
            b"hello again",
            b"goodbye world",
            b"goodbye there"
        ]
        y = [0, 0, 0, 1, 1]
        
        clf = CompressionKNN(n_neighbors=5)
        clf.fit(X, y)
        return clf
    
    def test_predict_proba_returns_list(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict_proba returns a list."""
        X_test = [b"hello"]
        probabilities = trained_classifier.predict_proba(X_test)
        
        assert isinstance(probabilities, list)
    
    def test_predict_proba_returns_float(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict_proba returns float values."""
        X_test = [b"hello"]
        probabilities = trained_classifier.predict_proba(X_test)
        
        assert isinstance(probabilities[0], (float, int))
    
    def test_predict_proba_range(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict_proba returns values in [0, 1] range."""
        X_test = [b"hello world", b"goodbye world"]
        probabilities = trained_classifier.predict_proba(X_test)
        
        for prob in probabilities:
            assert 0 <= prob <= 1
    
    def test_predict_proba_single_sample(self, trained_classifier: CompressionKNN) -> None:
        """Test predict_proba with single sample."""
        X_test = [b"hello"]
        probabilities = trained_classifier.predict_proba(X_test)
        
        assert len(probabilities) == 1
    
    def test_predict_proba_multiple_samples(self, trained_classifier: CompressionKNN) -> None:
        """Test predict_proba with multiple samples."""
        X_test = [b"hello", b"goodbye", b"test"]
        probabilities = trained_classifier.predict_proba(X_test)
        
        assert len(probabilities) == 3
    
    def test_predict_proba_high_probability_for_similar(self) -> None:
        """Test predict_proba returns high probability for very similar samples."""
        X_train = [b"aaa", b"aaa", b"aaa", b"bbb"]
        y_train = [0, 0, 0, 1]
        
        clf = CompressionKNN(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        probabilities = clf.predict_proba([b"aaa"])
        assert probabilities[0] >= 0.66  # At least 2/3 neighbors are class 0
    
    def test_predict_proba_with_k_neighbors(self) -> None:
        """Test predict_proba probability calculation with different k."""
        X_train = [b"a", b"a", b"a", b"b", b"b"]
        y_train = [0, 0, 0, 1, 1]
        
        clf = CompressionKNN(n_neighbors=5)
        clf.fit(X_train, y_train)
        
        # Test sample identical to class 0
        probabilities = clf.predict_proba([b"a"])
        # Should get 3/5 = 0.6 for class 0
        assert 0.5 <= probabilities[0] <= 0.7
    
    def test_predict_proba_consistency(self, trained_classifier: CompressionKNN) -> None:
        """Test that predict_proba returns consistent results for same input."""
        X_test = [b"hello world"]
        
        probs1 = trained_classifier.predict_proba(X_test)
        probs2 = trained_classifier.predict_proba(X_test)
        
        assert probs1 == probs2
    
    def test_predict_proba_empty_input(self, trained_classifier: CompressionKNN) -> None:
        """Test predict_proba with empty input list."""
        X_test = []
        probabilities = trained_classifier.predict_proba(X_test)
        
        assert probabilities == []
    
    def test_predict_proba_unanimous_vote(self) -> None:
        """Test predict_proba when all neighbors agree."""
        X_train = [b"cat1", b"cat2", b"cat3", b"dog1"]
        y_train = [0, 0, 0, 1]
        
        clf = CompressionKNN(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        # Sample very similar to cats
        probabilities = clf.predict_proba([b"cat1"])
        # All 3 nearest neighbors should be class 0
        assert probabilities[0] == 1.0 or probabilities[0] >= 0.66
    
    def test_predict_proba_with_string_inputs(self) -> None:
        """Test predict_proba automatically converts string inputs to bytes."""
        X_train = ["hello", "hello", "goodbye"]
        y_train = [0, 0, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        X_test = ["hello"]
        probabilities = clf.predict_proba(X_test)
        
        assert len(probabilities) == 1
        assert 0 <= probabilities[0] <= 1


class TestCompressionKNNEdgeCases:
    """Test edge cases and special scenarios for CompressionKNN."""
    
    def test_predict_before_fit_raises_error(self) -> None:
        """Test that predict before fit raises AttributeError."""
        clf = CompressionKNN()
        
        with pytest.raises(AttributeError):
            clf.predict([b"test"])
    
    def test_predict_proba_before_fit_raises_error(self) -> None:
        """Test that predict_proba before fit raises AttributeError."""
        clf = CompressionKNN()
        
        with pytest.raises(AttributeError):
            clf.predict_proba([b"test"])
    
    def test_n_neighbors_larger_than_training_set(self) -> None:
        """Test behavior when n_neighbors is larger than training set."""
        X_train = [b"sample1", b"sample2"]
        y_train = [0, 1]
        
        clf = CompressionKNN(n_neighbors=10)
        clf.fit(X_train, y_train)
        
        # Should still work, just using all available neighbors
        predictions = clf.predict([b"test"])
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    def test_single_training_sample_with_k_greater_than_one(self) -> None:
        """Test with single training sample and k > 1."""
        X_train = [b"only sample"]
        y_train = [42]
        
        clf = CompressionKNN(n_neighbors=5)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"test"])
        assert predictions[0] == 42
    
    def test_identical_training_samples(self) -> None:
        """Test with identical training samples."""
        X_train = [b"same", b"same", b"same"]
        y_train = [0, 0, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"same"])
        assert predictions[0] in [0, 1]
    
    def test_very_long_strings(self) -> None:
        """Test with very long strings."""
        X_train = [b"a" * 10000, b"b" * 10000]
        y_train = [0, 1]
        
        clf = CompressionKNN(n_neighbors=1)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"a" * 10000])
        assert predictions[0] == 0
    
    def test_empty_strings(self) -> None:
        """Test with empty strings."""
        X_train = [b"", b"data"]
        y_train = [0, 1]
        
        clf = CompressionKNN(n_neighbors=1)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b""])
        assert predictions[0] in [0, 1]
    
    def test_binary_data(self) -> None:
        """Test with binary data."""
        X_train = [bytes([i]) * 10 for i in range(5)]
        y_train = [0, 0, 1, 1, 1]
        
        clf = CompressionKNN(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([bytes([0]) * 10])
        assert predictions[0] in [0, 1]
    
    def test_multiclass_classification(self) -> None:
        """Test with more than 2 classes."""
        X_train = [
            b"class0 sample",
            b"class1 sample",
            b"class2 sample",
            b"class0 another",
            b"class1 another"
        ]
        y_train = [0, 1, 2, 0, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"class0 test"])
        assert predictions[0] in [0, 1, 2]
    
    def test_numeric_class_labels_not_sequential(self) -> None:
        """Test with non-sequential numeric labels."""
        X_train = [b"data1", b"data2", b"data3"]
        y_train = [10, 20, 10]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"data1"])
        assert predictions[0] in [10, 20]


class TestCompressionKNNIntegration:
    """Integration tests for CompressionKNN with realistic scenarios."""
    
    def test_text_classification_scenario(self) -> None:
        """Test CompressionKNN for text classification task."""
        # Train on simple spam vs ham messages
        X_train = [
            b"win free money now click here",
            b"get rich quick scheme offer",
            b"hello how are you doing today",
            b"meeting scheduled for tomorrow",
            b"buy now limited time offer",
            b"lunch plans for this week"
        ]
        y_train = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
        
        clf = CompressionKNN(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        # Test on new messages
        X_test = [
            b"free money offer",
            b"how are you"
        ]
        predictions = clf.predict(X_test)
        
        assert predictions[0] == 'spam'
        assert predictions[1] == 'ham'
    
    def test_language_detection_scenario(self) -> None:
        """Test CompressionKNN for language detection."""
        X_train = [
            b"hello world how are you",
            b"good morning everyone",
            b"bonjour le monde",
            b"comment allez vous",
            b"hola mundo como estas",
            b"buenos dias amigo"
        ]
        y_train = ['english', 'english', 'french', 'french', 'spanish', 'spanish']
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        X_test = [b"hello friend how are you doing"]
        predictions = clf.predict(X_test)
        
        assert predictions[0] == 'english'
    
    def test_dna_sequence_classification(self) -> None:
        """Test CompressionKNN with DNA-like sequences."""
        X_train = [
            b"ATGATGATGATG",
            b"ATGATCATGATC",
            b"GCGCGCGCGCGC",
            b"GCGAGCGAGCGA"
        ]
        y_train = [0, 0, 1, 1]
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"ATGATGATGATC"])
        assert predictions[0] == 0
        
        predictions = clf.predict([b"GCGCGCGCGCGA"])
        assert predictions[0] == 1
    
    def test_code_snippet_classification(self) -> None:
        """Test CompressionKNN for classifying code snippets."""
        X_train = [
            b"def function():\n    return value",
            b"def another():\n    return data",
            b"class MyClass:\n    def __init__(self):",
            b"class Another:\n    def __init__(self):"
        ]
        y_train = ['function', 'function', 'class', 'class']
        
        clf = CompressionKNN(n_neighbors=1)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict([b"def test():\n    return result"])
        assert predictions[0] == 'function'
    
    def test_high_dimensional_scenario(self) -> None:
        """Test CompressionKNN with larger dataset."""
        np.random.seed(42)
        
        # Generate synthetic data
        X_train = []
        y_train = []
        for i in range(50):
            if i < 25:
                X_train.append(b"pattern_A " * 10 + str(i).encode())
                y_train.append(0)
            else:
                X_train.append(b"pattern_B " * 10 + str(i).encode())
                y_train.append(1)
        
        clf = CompressionKNN(n_neighbors=5)
        clf.fit(X_train, y_train)
        
        X_test = [b"pattern_A " * 10 + b"test", b"pattern_B " * 10 + b"test"]
        predictions = clf.predict(X_test)
        
        assert predictions[0] == 0
        assert predictions[1] == 1
    
    def test_imbalanced_classes(self) -> None:
        """Test CompressionKNN with imbalanced class distribution."""
        X_train = [b"majority"] * 9 + [b"minority"]
        y_train = [0] * 9 + [1]
        
        clf = CompressionKNN(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        # Should predict majority class for similar sample
        predictions = clf.predict([b"majority"])
        assert predictions[0] == 0
    
    def test_multilingual_text_with_encoding(self) -> None:
        """Test CompressionKNN with multilingual text using appropriate encoding."""
        X_train = [
            "café résumé",
            "naïve jalapeño",
            "hello world",
            "goodbye earth"
        ]
        y_train = ['french', 'french', 'english', 'english']
        
        clf = CompressionKNN(n_neighbors=2, encoding='utf-8')
        clf.fit(X_train, y_train)
        
        X_test = ["crème brûlée", "computer science"]
        predictions = clf.predict(X_test)
        
        # Check predictions are valid
        assert predictions[0] in ['french', 'english']
        assert predictions[1] in ['french', 'english']
    
    def test_string_based_workflow(self) -> None:
        """Test complete workflow using only strings (no bytes)."""
        # This tests the convenience of automatic conversion
        X_train = [
            "apple fruit red",
            "banana fruit yellow",
            "carrot vegetable orange",
            "broccoli vegetable green"
        ]
        y_train = ['fruit', 'fruit', 'vegetable', 'vegetable']
        
        clf = CompressionKNN(n_neighbors=2)
        clf.fit(X_train, y_train)
        
        X_test = ["orange fruit", "spinach vegetable"]
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        assert len(predictions) == 2
        assert len(probabilities) == 2
        assert all(p in ['fruit', 'vegetable'] for p in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)
