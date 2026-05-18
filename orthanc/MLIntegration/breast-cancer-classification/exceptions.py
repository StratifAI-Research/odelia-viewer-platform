"""
Service-specific exceptions for Breast Cancer Classification
"""


class ModelNotLoadedError(Exception):
    """Raised when model is not loaded but inference is requested"""
    pass


class InferenceError(Exception):
    """Raised when model inference fails"""
    pass


class PreprocessingError(Exception):
    """Raised when preprocessing fails"""
    pass
