"""Sibling module names shared across ML services.

Any of these names in sys.modules from a prior test's import will resolve to
the wrong service in the next test.  Per-service conftests import this tuple
and evict the matching entries from sys.modules between tests.
"""

ML_SERVICE_COLLIDERS = (
    'app', 'app_refactored', 'benchmark', 'config',
    'debug_routes', 'dicom_converter', 'dicom_utils',
    'dicom2nfti_onthefly', 'exceptions', 'image_cache',
    'models', 'model_loader', 'model_service',
    'ollama_client', 'preprocessing', 'prompt_builder',
    'prompt_templates', 'response_builder', 'response_parser',
    'retrieval_strategy', 'runtime_config', 'session_manager',
    'websocket_handler',
)
