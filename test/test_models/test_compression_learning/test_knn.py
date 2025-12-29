"""Unit tests for CompressionKNN classifier."""

import pytest
import pandas as pd
import polars as pl
import narwhals as nw
from explodingham.models.compression_learning.knn import CompressionKNN, BaseKNNModel


class TestBaseKNNModelInitialization:
    """Test suite for BaseKNNModel initialization."""
    
    def test_initialization_with_k(self) -> None:
        """Test BaseKNNModel initializes with k parameter."""
        model = BaseKNNModel(k=5)
        assert model.k == 5
    
    def test_initialization_with_different_k_values(self) -> None:
        """Test BaseKNNModel with various k values."""
        for k in [1, 3, 10, 100]:
            model = BaseKNNModel(k=k)
            assert model.k == k


class TestCompressionKNNInitialization:
    """Test suite for CompressionKNN initialization."""
    
    def test_default_initialization(self) -> None:
        """Test CompressionKNN initializes with default parameters."""
        clf = CompressionKNN(k=5)
        assert clf.k == 5
        assert clf.encoding == 'utf-8'
        assert clf.encoded is False
        assert clf.data_column is None
    
    def test_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        clf = CompressionKNN(
            k=3,
            data_column='text',
            encoding='latin-1',
            compressor='bz2',
            encoded=True
        )
        assert clf.k == 3
        assert clf.data_column == 'text'
        assert clf.encoding == 'latin-1'
        assert clf.encoded is True
    
    def test_gzip_compressor_from_string(self) -> None:
        """Test gzip compressor is loaded from string."""
        clf = CompressionKNN(k=5, compressor='gzip')
        import gzip
        assert clf.compressor == gzip.compress
    
    def test_bz2_compressor_from_string(self) -> None:
        """Test bz2 compressor is loaded from string."""
        clf = CompressionKNN(k=5, compressor='bz2')
        import bz2
        assert clf.compressor == bz2.compress
    
    def test_lzma_compressor_from_string(self) -> None:
        """Test lzma compressor is loaded from string."""
        clf = CompressionKNN(k=5, compressor='lzma')
        import lzma
        assert clf.compressor == lzma.compress
    
    def test_callable_compressor(self) -> None:
        """Test initialization with callable compressor."""
        import gzip
        clf = CompressionKNN(k=5, compressor=gzip.compress)
        assert clf.compressor == gzip.compress
    
    def test_invalid_compressor_raises_error(self) -> None:
        """Test initialization with invalid compressor string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CompressionKNN(k=5, compressor='invalid')
        assert "Unsupported compressor" in str(exc_info.value)


class TestCompressionKNNFit:
    """Test suite for CompressionKNN.fit method."""
    
    @pytest.fixture
    def simple_pandas_data(self) -> tuple:
        """Create simple training data as pandas DataFrame."""
        X = pd.DataFrame({
            'text': ['hello world', 'hello there', 'goodbye world', 'goodbye there']
        })
        y = pd.Series([0, 0, 1, 1], name='label')
        return X, y
    
    @pytest.fixture
    def simple_polars_data(self) -> tuple:
        """Create simple training data as polars DataFrame."""
        X = pl.DataFrame({
            'text': ['hello world', 'hello there', 'goodbye world', 'goodbye there']
        })
        y = pl.Series('label', [0, 0, 1, 1])
        return X, y
    
    def test_fit_with_pandas_dataframe(self, simple_pandas_data: tuple) -> None:
        """Test fit with pandas DataFrame."""
        X, y = simple_pandas_data
        clf = CompressionKNN(k=2, data_column='text')
        result = clf.fit(X, y)
        
        assert hasattr(clf, 'model_data')
        assert hasattr(clf, 'target_column')
        assert clf.target_column == 'label'
        assert result is clf
    
    def test_fit_with_polars_dataframe(self, simple_polars_data: tuple) -> None:
        """Test fit with polars DataFrame."""
        X, y = simple_polars_data
        clf = CompressionKNN(k=2, data_column='text')
        result = clf.fit(X, y)
        
        assert hasattr(clf, 'model_data')
        assert hasattr(clf, 'target_column')
        assert clf.target_column == 'label'
        assert result is clf
    
    def test_fit_with_pandas_series(self) -> None:
        """Test fit with pandas Series as X."""
        X = pd.Series(['hello', 'goodbye', 'hello', 'goodbye'], name='text')
        y = pd.Series([0, 1, 0, 1], name='label')
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        
        assert hasattr(clf, 'model_data')
        assert clf.target_column == 'label'
    
    def test_fit_stores_backend(self, simple_pandas_data: tuple) -> None:
        """Test fit stores the backend implementation."""
        X, y = simple_pandas_data
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        
        assert hasattr(clf, 'backend')
    
    def test_fit_with_string_labels(self) -> None:
        """Test fit with string class labels."""
        X = pd.DataFrame({
            'text': ['cat meow', 'dog bark', 'cat purr']
        })
        y = pd.Series(['cat', 'dog', 'cat'], name='animal')
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        
        assert clf.target_column == 'animal'
    
    def test_fit_with_multicolumn_y_raises_error(self) -> None:
        """Test fit with multi-column y DataFrame raises ValueError."""
        X = pd.DataFrame({'text': ['a', 'b']})
        y = pd.DataFrame({'label1': [0, 1], 'label2': [2, 3]})
        
        clf = CompressionKNN(k=1, data_column='text')
        
        with pytest.raises(ValueError) as exc_info:
            clf.fit(X, y)
        assert "exactly one column" in str(exc_info.value)


class TestCompressionKNNPredict:
    """Test suite for CompressionKNN.predict method."""
    
    @pytest.fixture
    def trained_pandas_classifier(self) -> CompressionKNN:
        """Create a trained classifier with pandas data."""
        X = pd.DataFrame({
            'text': [
                'hello world hello world',
                'hello there hello there',
                'goodbye world goodbye world',
                'goodbye there goodbye there'
            ]
        })
        y = pd.Series([0, 0, 1, 1], name='label')
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        return clf
    
    @pytest.fixture
    def trained_polars_classifier(self) -> CompressionKNN:
        """Create a trained classifier with polars data."""
        X = pl.DataFrame({
            'text': [
                'hello world hello world',
                'hello there hello there',
                'goodbye world goodbye world',
                'goodbye there goodbye there'
            ]
        })
        y = pl.Series('label', [0, 0, 1, 1])
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        return clf
    
    def test_predict_returns_grouped_dataframe(self, trained_pandas_classifier: CompressionKNN) -> None:
        """Test predict returns a grouped DataFrame."""
        X_test = pd.DataFrame({'text': ['hello world']})
        result = trained_pandas_classifier.predict(X_test)
        
        # Result should be a narwhals GroupBy object or similar
        assert result is not None
    
    def test_predict_with_multiple_samples(self, trained_pandas_classifier: CompressionKNN) -> None:
        """Test predict with multiple samples."""
        X_test = pd.DataFrame({'text': ['hello world', 'goodbye world']})
        result = trained_pandas_classifier.predict(X_test)
        
        assert result is not None
    
    def test_predict_with_polars(self, trained_polars_classifier: CompressionKNN) -> None:
        """Test predict with polars DataFrame."""
        X_test = pl.DataFrame({'text': ['hello world']})
        result = trained_polars_classifier.predict(X_test)
        
        assert result is not None
    
    def test_predict_with_series(self, trained_pandas_classifier: CompressionKNN) -> None:
        """Test predict with pandas Series."""
        X_test = pd.Series(['hello world', 'goodbye world'], name='text')
        result = trained_pandas_classifier.predict(X_test)
        
        assert result is not None


class TestCompressionKNNHelperMethods:
    """Test suite for CompressionKNN helper methods."""
    
    def test_handle_X_converts_series_to_dataframe(self) -> None:
        """Test _handle_X converts Series to DataFrame."""
        clf = CompressionKNN(k=5)
        X = pd.Series(['a', 'b', 'c'], name='text')
        
        result = clf._handle_X(X)
        
        # Should be converted to DataFrame
        result_native = nw.from_native(result)
        assert isinstance(result_native.to_native(), (pd.DataFrame, pl.DataFrame))
    
    def test_handle_X_preserves_dataframe(self) -> None:
        """Test _handle_X preserves DataFrame input."""
        clf = CompressionKNN(k=5)
        X = pd.DataFrame({'text': ['a', 'b', 'c']})
        
        result = clf._handle_X(X)
        
        result_native = nw.from_native(result)
        assert isinstance(result_native.to_native(), (pd.DataFrame, pl.DataFrame))
    
    def test_get_callable_compressor_gzip(self) -> None:
        """Test _get_callable_compressor returns gzip.compress."""
        clf = CompressionKNN(k=5)
        import gzip
        
        compressor = clf._get_callable_compressor('gzip')
        assert compressor == gzip.compress
    
    def test_get_callable_compressor_bz2(self) -> None:
        """Test _get_callable_compressor returns bz2.compress."""
        clf = CompressionKNN(k=5)
        import bz2
        
        compressor = clf._get_callable_compressor('bz2')
        assert compressor == bz2.compress
    
    def test_get_callable_compressor_lzma(self) -> None:
        """Test _get_callable_compressor returns lzma.compress."""
        clf = CompressionKNN(k=5)
        import lzma
        
        compressor = clf._get_callable_compressor('lzma')
        assert compressor == lzma.compress
    
    def test_get_callable_compressor_invalid_raises_error(self) -> None:
        """Test _get_callable_compressor raises ValueError for invalid compressor."""
        clf = CompressionKNN(k=5)
        
        with pytest.raises(ValueError) as exc_info:
            clf._get_callable_compressor('invalid_compressor')
        assert "Unsupported compressor" in str(exc_info.value)


class TestCompressionKNNIntegration:
    """Integration tests for CompressionKNN with realistic scenarios."""
    
    def test_text_classification_pandas(self) -> None:
        """Test CompressionKNN for text classification with pandas."""
        X_train = pd.DataFrame({
            'text': [
                'win free money now click here',
                'get rich quick scheme offer',
                'hello how are you doing today',
                'meeting scheduled for tomorrow',
                'buy now limited time offer',
                'lunch plans for this week'
            ]
        })
        y_train = pd.Series(['spam', 'spam', 'ham', 'ham', 'spam', 'ham'], name='category')
        
        clf = CompressionKNN(k=3, data_column='text')
        clf.fit(X_train, y_train)
        
        X_test = pd.DataFrame({
            'text': ['free money offer', 'how are you']
        })
        result = clf.predict(X_test)
        
        # Result should be a grouped structure
        assert result is not None
    
    def test_text_classification_polars(self) -> None:
        """Test CompressionKNN for text classification with polars."""
        X_train = pl.DataFrame({
            'text': [
                'win free money now click here',
                'get rich quick scheme offer',
                'hello how are you doing today',
                'meeting scheduled for tomorrow'
            ]
        })
        y_train = pl.Series('category', ['spam', 'spam', 'ham', 'ham'])
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X_train, y_train)
        
        X_test = pl.DataFrame({'text': ['free money offer']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_multiclass_classification(self) -> None:
        """Test with more than 2 classes."""
        X_train = pd.DataFrame({
            'text': [
                'python programming language',
                'java programming language',
                'apple fruit red',
                'banana fruit yellow',
                'car vehicle transportation',
                'bus vehicle transportation'
            ]
        })
        y_train = pd.Series(
            ['programming', 'programming', 'food', 'food', 'transport', 'transport'],
            name='category'
        )
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X_train, y_train)
        
        X_test = pd.DataFrame({'text': ['javascript coding']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_with_different_compressors(self) -> None:
        """Test that different compressors work."""
        X = pd.DataFrame({'text': ['a' * 100, 'b' * 100, 'a' * 100]})
        y = pd.Series([0, 1, 0], name='label')
        
        for compressor_name in ['gzip', 'bz2', 'lzma']:
            clf = CompressionKNN(k=2, data_column='text', compressor=compressor_name)
            clf.fit(X, y)
            
            X_test = pd.DataFrame({'text': ['a' * 100]})
            result = clf.predict(X_test)
            
            assert result is not None


class TestCompressionKNNEdgeCases:
    """Test edge cases and special scenarios for CompressionKNN."""
    
    def test_single_training_sample(self) -> None:
        """Test with single training sample."""
        X = pd.DataFrame({'text': ['only sample']})
        y = pd.Series([42], name='label')
        
        clf = CompressionKNN(k=5, data_column='text')
        clf.fit(X, y)
        
        X_test = pd.DataFrame({'text': ['test']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_k_larger_than_training_set(self) -> None:
        """Test behavior when k is larger than training set."""
        X = pd.DataFrame({'text': ['sample1', 'sample2']})
        y = pd.Series([0, 1], name='label')
        
        clf = CompressionKNN(k=10, data_column='text')
        clf.fit(X, y)
        
        X_test = pd.DataFrame({'text': ['test']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_empty_strings(self) -> None:
        """Test with empty strings."""
        X = pd.DataFrame({'text': ['', 'data', '']})
        y = pd.Series([0, 1, 0], name='label')
        
        clf = CompressionKNN(k=2, data_column='text')
        clf.fit(X, y)
        
        X_test = pd.DataFrame({'text': ['']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_unicode_text(self) -> None:
        """Test with Unicode text."""
        X = pd.DataFrame({'text': ['café résumé', 'naïve jalapeño', 'hello world']})
        y = pd.Series(['french', 'french', 'english'], name='language')
        
        clf = CompressionKNN(k=2, data_column='text', encoding='utf-8')
        clf.fit(X, y)
        
        X_test = pd.DataFrame({'text': ['crème brûlée']})
        result = clf.predict(X_test)
        
        assert result is not None
    
    def test_very_long_strings(self) -> None:
        """Test with very long strings."""
        X = pd.DataFrame({'text': ['a' * 10000, 'b' * 10000]})
        y = pd.Series([0, 1], name='label')
        
        clf = CompressionKNN(k=1, data_column='text')
        clf.fit(X, y)
        
        X_test = pd.DataFrame({'text': ['a' * 10000]})
        result = clf.predict(X_test)
        
        assert result is not None
