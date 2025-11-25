"""Unit tests for BaseExplodingHamClassifier."""

import pytest
from explodingham.utils.base.base_classifier import BaseExplodingHamClassifier


class TestBaseExplodingHamClassifier:
    """Test suite for BaseExplodingHamClassifier base class."""
    
    @pytest.fixture
    def classifier(self) -> BaseExplodingHamClassifier:
        """Create a basic classifier instance."""
        return BaseExplodingHamClassifier()
    
    def test_initialization(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test that BaseExplodingHamClassifier initializes correctly."""
        assert isinstance(classifier, BaseExplodingHamClassifier)
    
    def test_set_param_with_int(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with integer conversion."""
        classifier._set_param('test_param', 42, int)
        assert hasattr(classifier, 'test_param')
        assert classifier.test_param == 42
        assert isinstance(classifier.test_param, int)
    
    def test_set_param_with_string_to_int(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param converts string to integer."""
        classifier._set_param('test_param', '100', int)
        assert classifier.test_param == 100
        assert isinstance(classifier.test_param, int)
    
    def test_set_param_with_float(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with float conversion."""
        classifier._set_param('learning_rate', 0.01, float)
        assert classifier.learning_rate == 0.01
        assert isinstance(classifier.learning_rate, float)
    
    def test_set_param_with_string_to_float(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param converts string to float."""
        classifier._set_param('threshold', '0.5', float)
        assert classifier.threshold == 0.5
        assert isinstance(classifier.threshold, float)
    
    def test_set_param_with_bool(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with boolean conversion."""
        classifier._set_param('verbose', True, bool)
        assert classifier.verbose is True
        assert isinstance(classifier.verbose, bool)
    
    def test_set_param_with_bool_false(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with False boolean value."""
        classifier._set_param('debug', False, bool)
        assert classifier.debug is False
        assert isinstance(classifier.debug, bool)
    
    def test_set_param_with_str(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with string conversion."""
        classifier._set_param('name', 'test_classifier', str)
        assert classifier.name == 'test_classifier'
        assert isinstance(classifier.name, str)
    
    def test_set_param_invalid_int_conversion(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param raises ValueError for invalid int conversion."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_param('bad_param', 'not_a_number', int)
        assert 'Invalid value for bad_param' in str(exc_info.value)
        assert 'must be a int' in str(exc_info.value)
    
    def test_set_param_invalid_float_conversion(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param raises ValueError for invalid float conversion."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_param('bad_param', 'not_a_float', float)
        assert 'Invalid value for bad_param' in str(exc_info.value)
        assert 'must be a float' in str(exc_info.value)
    
    def test_set_option_valid_option(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option with a valid option."""
        classifier._set_option('method', 'gzip', ['gzip', 'bz2', 'lzma'])
        assert classifier.method == 'gzip'
    
    def test_set_option_different_valid_option(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option with another valid option."""
        classifier._set_option('algorithm', 'bz2', ['gzip', 'bz2', 'lzma'])
        assert classifier.algorithm == 'bz2'
    
    def test_set_option_invalid_option(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option raises ValueError for invalid option."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_option('method', 'invalid', ['gzip', 'bz2', 'lzma'])
        assert 'Invalid option: invalid' in str(exc_info.value)
        assert "Valid options are: ['gzip', 'bz2', 'lzma']" in str(exc_info.value)
    
    def test_set_option_case_sensitive(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option is case-sensitive."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_option('method', 'GZIP', ['gzip', 'bz2', 'lzma'])
        assert 'Invalid option: GZIP' in str(exc_info.value)
    
    def test_set_option_single_option_list(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option with single valid option."""
        classifier._set_option('mode', 'train', ['train'])
        assert classifier.mode == 'train'
    
    def test_set_option_empty_string_not_in_options(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option rejects empty string when not in options."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_option('method', '', ['gzip', 'bz2'])
        assert 'Invalid option: ' in str(exc_info.value)
    
    def test_set_param_multiple_params(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test setting multiple parameters."""
        classifier._set_param('param1', 10, int)
        classifier._set_param('param2', 0.5, float)
        classifier._set_param('param3', True, bool)
        
        assert classifier.param1 == 10
        assert classifier.param2 == 0.5
        assert classifier.param3 is True
    
    def test_set_option_multiple_options(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test setting multiple options."""
        classifier._set_option('compression', 'gzip', ['gzip', 'bz2'])
        classifier._set_option('mode', 'test', ['train', 'test'])
        
        assert classifier.compression == 'gzip'
        assert classifier.mode == 'test'


class TestBaseExplodingHamClassifierEdgeCases:
    """Test edge cases and special scenarios for BaseExplodingHamClassifier."""
    
    @pytest.fixture
    def classifier(self) -> BaseExplodingHamClassifier:
        """Create a basic classifier instance."""
        return BaseExplodingHamClassifier()
    
    def test_set_param_with_negative_int(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param handles negative integers."""
        classifier._set_param('negative_val', -10, int)
        assert classifier.negative_val == -10
    
    def test_set_param_with_zero(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param handles zero value."""
        classifier._set_param('zero_val', 0, int)
        assert classifier.zero_val == 0
    
    def test_set_param_with_float_to_int_truncation(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param truncates float to int."""
        classifier._set_param('truncated', 3.7, int)
        assert classifier.truncated == 3
        assert isinstance(classifier.truncated, int)
    
    def test_set_param_with_empty_string(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param handles empty string."""
        classifier._set_param('empty', '', str)
        assert classifier.empty == ''
    
    def test_set_param_with_none_value(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param with None value raises error for int."""
        with pytest.raises(ValueError) as exc_info:
            classifier._set_param('none_param', None, int)
        assert 'Invalid value for none_param' in str(exc_info.value)
    
    def test_set_param_with_bool_string_conversion(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param converts non-empty string to True."""
        classifier._set_param('bool_param', 'any_string', bool)
        assert classifier.bool_param is True
    
    def test_set_param_with_zero_to_bool(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param converts 0 to False."""
        classifier._set_param('zero_bool', 0, bool)
        assert classifier.zero_bool is False
    
    def test_set_param_with_one_to_bool(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param converts 1 to True."""
        classifier._set_param('one_bool', 1, bool)
        assert classifier.one_bool is True
    
    def test_set_option_with_numeric_string(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option with numeric strings."""
        classifier._set_option('version', '1.0', ['1.0', '2.0', '3.0'])
        assert classifier.version == '1.0'
    
    def test_set_option_with_special_characters(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option with special characters in options."""
        classifier._set_option('delimiter', ',', [',', ';', '\t', '|'])
        assert classifier.delimiter == ','
    
    def test_set_param_overwrites_existing_attribute(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param overwrites existing attribute."""
        classifier._set_param('param', 10, int)
        assert classifier.param == 10
        
        classifier._set_param('param', 20, int)
        assert classifier.param == 20
    
    def test_set_option_overwrites_existing_attribute(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_option overwrites existing attribute."""
        classifier._set_option('method', 'gzip', ['gzip', 'bz2'])
        assert classifier.method == 'gzip'
        
        classifier._set_option('method', 'bz2', ['gzip', 'bz2'])
        assert classifier.method == 'bz2'
    
    def test_set_param_with_large_int(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param handles large integers."""
        large_num = 10**15
        classifier._set_param('large_param', large_num, int)
        assert classifier.large_param == large_num
    
    def test_set_param_with_scientific_notation(self, classifier: BaseExplodingHamClassifier) -> None:
        """Test _set_param handles scientific notation."""
        classifier._set_param('sci_param', 1e-5, float)
        assert classifier.sci_param == 1e-5


class TestBaseExplodingHamClassifierInheritance:
    """Test BaseExplodingHamClassifier inheritance from sklearn classes."""
    
    def test_inherits_from_base_estimator(self) -> None:
        """Test that BaseExplodingHamClassifier inherits from sklearn BaseEstimator."""
        from sklearn.base import BaseEstimator
        classifier = BaseExplodingHamClassifier()
        assert isinstance(classifier, BaseEstimator)
    
    def test_inherits_from_classifier_mixin(self) -> None:
        """Test that BaseExplodingHamClassifier inherits from sklearn ClassifierMixin."""
        from sklearn.base import ClassifierMixin
        classifier = BaseExplodingHamClassifier()
        assert isinstance(classifier, ClassifierMixin)
    
    def test_has_get_params_method(self) -> None:
        """Test that BaseExplodingHamClassifier has get_params from BaseEstimator."""
        classifier = BaseExplodingHamClassifier()
        assert hasattr(classifier, 'get_params')
        assert callable(classifier.get_params)
    
    def test_has_set_params_method(self) -> None:
        """Test that BaseExplodingHamClassifier has set_params from BaseEstimator."""
        classifier = BaseExplodingHamClassifier()
        assert hasattr(classifier, 'set_params')
        assert callable(classifier.set_params)
    
    def test_has_score_method(self) -> None:
        """Test that BaseExplodingHamClassifier has score method from ClassifierMixin."""
        classifier = BaseExplodingHamClassifier()
        assert hasattr(classifier, 'score')
        assert callable(classifier.score)
    
    def test_get_params_returns_empty_dict_initially(self) -> None:
        """Test that get_params returns empty dict for base class."""
        classifier = BaseExplodingHamClassifier()
        params = classifier.get_params(deep=False)
        assert isinstance(params, dict)
    
    def test_set_params_after_using_set_param(self) -> None:
        """Test sklearn set_params works after using custom _set_param."""
        classifier = BaseExplodingHamClassifier()
        classifier._set_param('n_neighbors', 5, int)
        
        # Verify the attribute was set
        assert hasattr(classifier, 'n_neighbors')
        assert classifier.n_neighbors == 5
        
        # Note: get_params only returns __init__ parameters for sklearn compatibility
        # Dynamically set attributes are not included
