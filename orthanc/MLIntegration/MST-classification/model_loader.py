"""
HuggingFace model download and loading logic for MST model
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, login
from model_integrity import read_integrity, verify_files

logger = logging.getLogger(__name__)

# Configuration
MODEL_REPO = "ODELIA-AI/MST"
MODEL_PATH = os.getenv("MODEL_PATH", "./mst_model")
HF_TOKEN = os.getenv("HF_TOKEN", None)
HTTP_PROXY = os.getenv("HTTP_PROXY", None)
HTTPS_PROXY = os.getenv("HTTPS_PROXY", None)


def get_proxy_config() -> dict[str, str] | None:
    """Get proxy configuration dict for HuggingFace requests only"""
    proxies = {}
    if HTTP_PROXY:
        proxies["http"] = HTTP_PROXY
        proxies["https"] = HTTPS_PROXY if HTTPS_PROXY else HTTP_PROXY
        logger.info(f"Using proxy for HuggingFace downloads: {HTTP_PROXY}")
    return proxies if proxies else None


def _clear_proxy_settings() -> None:
    # Keep download-only proxies out of subsequent internal PACS requests.
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(name, None)
    logger.info("Proxy settings cleared after download")


def download_model_files() -> dict[str, str]:
    """
    Download required model files from HuggingFace

    Returns:
        dict: Paths to downloaded files
    """

    integrity = read_integrity(Path(__file__).with_name("model-integrity.json"))
    if all((Path(MODEL_PATH) / name).is_file() for name in integrity["sha256"]):
        try:
            return verify_files(Path(MODEL_PATH), integrity)
        finally:
            _clear_proxy_settings()

    logger.info(f"Downloading MST model files from {MODEL_REPO}")

    # Get proxy config for HuggingFace only
    proxies = get_proxy_config()

    # Configure httpx client with proxy if needed
    if proxies:
        import huggingface_hub

        # Set proxy only for HuggingFace operations
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

    # Authenticate with HuggingFace if token provided
    if HF_TOKEN:
        logger.info("Authenticating with HuggingFace token")
        login(token=HF_TOKEN)
    else:
        logger.info(
            "No HF_TOKEN provided - proceeding unauthenticated (fine for public repos like ODELIA-AI/MST)"
        )

    # Create model directory
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # Download required files
    files_to_download = [
        "models.py",
        "predict_attention.py",
        "model_config.json",
        "state_dict.pt",  # The actual model weights
    ]

    downloaded_files = {}

    # Temporarily set proxy env vars only for HuggingFace downloads
    try:
        if proxies:
            os.environ["HTTP_PROXY"] = proxies.get("http", "")
            os.environ["HTTPS_PROXY"] = proxies.get("https", "")
            os.environ["http_proxy"] = proxies.get("http", "")
            os.environ["https_proxy"] = proxies.get("https", "")

        for filename in files_to_download:
            try:
                logger.info(f"Downloading {filename}...")
                file_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=MODEL_PATH,
                    token=HF_TOKEN,
                    revision=integrity["revision"],
                )
                downloaded_files[filename] = file_path
                logger.info(f"✓ Downloaded {filename} to {file_path}")
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                raise
    finally:
        _clear_proxy_settings()

    logger.info("All model files downloaded successfully")
    return verify_files(Path(MODEL_PATH), integrity)


def load_model() -> tuple[Any, Any, dict[str, str]]:
    """
    Load the MST model

    Returns:
        tuple: (model, predict_function, model_info)
    """
    try:
        integrity = read_integrity(Path(__file__).with_name("model-integrity.json"))
        files = verify_files(Path(MODEL_PATH), integrity)

        # Add model path to Python path to import downloaded modules
        sys.path.insert(0, MODEL_PATH)

        # Use ordinary imports after verifying the downloaded bundle.
        import torch
        from models import MSTRegression
        from predict_attention import run_prediction

        logger.info("Loading MST model from verified local files...")
        config = json.loads(Path(files["model_config.json"]).read_text())
        model = MSTRegression(weights=False, **config.get("hparams", {}))
        state_dict = torch.load(files["state_dict.pt"], map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        logger.info("✓ MST model loaded successfully")

        model_info = {
            "model_name": "ODELIA-AI",
            "architecture": "Vision Transformer",
            "version": "1.0",
            "revision": integrity["revision"],
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
