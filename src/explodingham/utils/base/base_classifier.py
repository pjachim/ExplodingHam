from sklearn.base import BaseEstimator, ClassifierMixin

class BaseExplodingHamClassifier(BaseEstimator, ClassifierMixin):
    def _set_option(self, param_name: str, value: str, options: list[str]) -> None:
        if value not in options:
            raise ValueError(f"Invalid option: {value}. Valid options are: {options}")
        
        setattr(self, param_name, value)

    def _set_param(self, param, value, type_conversion_func) -> None:
        try:
            setattr(self, param, type_conversion_func(value))
        except Exception as e:
            raise ValueError(f"Invalid value for {param}, must be a {type_conversion_func.__name__}: {value}.\n\nError:\n{e}")
        
