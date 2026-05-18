"""
Service-specific exceptions for MedGemma MRI Classification
"""


class ModelNotLoadedError(Exception):
    """Raised when model is not loaded but inference is requested"""

    def __init__(self, message: str = "Model not loaded"):
        self.message = message
        super().__init__(self.message)


class ModelAuthenticationError(Exception):
    """Raised when HuggingFace authentication fails (invalid token or license not accepted)"""

    def __init__(self, message: str = "HuggingFace authentication failed"):
        self.message = message
        super().__init__(self.message)


class InferenceError(Exception):
    """Raised when model inference fails"""

    def __init__(self, message: str = "Model inference failed"):
        self.message = message
        super().__init__(self.message)


class PreprocessingError(Exception):
    """Raised when DICOM preprocessing or slice extraction fails"""

    def __init__(self, message: str = "Preprocessing failed"):
        self.message = message
        super().__init__(self.message)


class ResponseParsingError(Exception):
    """Raised when parsing MedGemma JSON response fails"""

    def __init__(self, message: str = "Failed to parse model response", raw_response: str = None):
        self.message = message
        self.raw_response = raw_response
        super().__init__(self.message)
