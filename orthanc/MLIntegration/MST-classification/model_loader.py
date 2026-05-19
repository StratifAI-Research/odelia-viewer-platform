"""
HuggingFace model download and loading logic for MST model
"""
import os
import sys
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, login

logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO = "ODELIA-AI/MST"
MODEL_PATH = os.getenv("MODEL_PATH", "./mst_model")
HF_TOKEN = os.getenv("HF_TOKEN", None)
HTTP_PROXY = os.getenv("HTTP_PROXY", None)
HTTPS_PROXY = os.getenv("HTTPS_PROXY", None)

def get_proxy_config():
    """Get proxy configuration dict for HuggingFace requests only"""
    proxies = {}
    if HTTP_PROXY:
        proxies['http'] = HTTP_PROXY
        proxies['https'] = HTTPS_PROXY if HTTPS_PROXY else HTTP_PROXY
        logger.info(f"Using proxy for HuggingFace downloads: {HTTP_PROXY}")
    return proxies if proxies else None

def download_model_files():
    """
    Download required model files from HuggingFace

    Returns:
        dict: Paths to downloaded files
    """
    import httpx

    logger.info(f"Downloading MST model files from {MODEL_REPO}")

    # Get proxy config for HuggingFace only
    proxies = get_proxy_config()

    # Configure httpx client with proxy if needed
    if proxies:
        import huggingface_hub
        # Set proxy only for HuggingFace operations
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'

    # Authenticate with HuggingFace if token provided
    if HF_TOKEN:
        logger.info("Authenticating with HuggingFace token")
        login(token=HF_TOKEN)
    else:
        logger.warning("No HF_TOKEN provided - may fail for gated models")

    # Create model directory
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Download required files
    files_to_download = [
        "models.py",
        "predict_attention.py",
        "model_config.json",
        "state_dict.pt"  # The actual model weights
    ]

    downloaded_files = {}

    # Temporarily set proxy env vars only for HuggingFace downloads
    try:
        if proxies:
            os.environ['HTTP_PROXY'] = proxies.get('http', '')
            os.environ['HTTPS_PROXY'] = proxies.get('https', '')
            os.environ['http_proxy'] = proxies.get('http', '')
            os.environ['https_proxy'] = proxies.get('https', '')

        for filename in files_to_download:
            try:
                logger.info(f"Downloading {filename}...")
                file_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=MODEL_PATH,
                    token=HF_TOKEN
                )
                downloaded_files[filename] = file_path
                logger.info(f"✓ Downloaded {filename} to {file_path}")
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                raise
    finally:
        # Remove ALL proxy settings after download completes
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('https_proxy', None)
        logger.info("Proxy settings cleared after download")

    logger.info("All model files downloaded successfully")
    return downloaded_files


def load_model():
    """
    Load the MST model

    Returns:
        tuple: (model, predict_function, model_info)
    """
    try:
        # Add model path to Python path to import downloaded modules
        sys.path.insert(0, MODEL_PATH)

        # Import the downloaded modules
        from predict_attention import load_model as load_mst_model, run_prediction

        logger.info("Loading MST model...")

        # Load model using the provided function
        model = load_mst_model(repo_id="ODELIA-AI/MST")
        model.eval()

        logger.info("✓ MST model loaded successfully")

        model_info = {
            "model_name": "ODELIA-AI",
            "architecture": "Vision Transformer",
            "version": "1.0"
        }

        return model, run_prediction, model_info

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


if __name__ == "__main__":
    # Test model download
    logging.basicConfig(level=logging.INFO)
    download_model_files()
    model, predict_fn, info = load_model()
    print(f"Model loaded: {info}")
