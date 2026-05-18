"""
FastAPI application entry point for the chat middleware service
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import after logging is configured
from config import init_config, get_config
from runtime_config import get_runtime_config
from session_manager import get_session_manager
from image_cache import get_image_cache
from ollama_client import get_ollama_client
from websocket_handler import handle_websocket
from debug_routes import router as debug_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Chat Middleware Service - Starting")
    logger.info("=" * 60)

    # Initialize configuration
    config = init_config()
    logger.info(f"Backend type: {config.backend_type}")
    logger.info(f"LLM URL: {config.ollama_url}")
    logger.info(f"LLM Model: {config.ollama_model}")
    logger.info(f"WADO-RS Base URL: {config.wado_base_url}")
    logger.info(f"Image folder: {config.image_folder}")
    logger.info(f"Max cache entries: {config.max_cache_entries}")

    # Initialize singletons
    get_runtime_config()
    get_session_manager()
    get_image_cache(max_entries=config.max_cache_entries)

    # Initialize Ollama client and check connectivity
    ollama_client = get_ollama_client()
    ollama_healthy = await ollama_client.health_check()
    if ollama_healthy:
        logger.info("Ollama connection: OK")
        models = await ollama_client.list_models()
        logger.info(f"Available models: {models}")
    else:
        logger.warning("Ollama connection: FAILED - service may be unavailable")

    logger.info("=" * 60)
    logger.info(f"Service ready on http://{config.host}:{config.port}")
    logger.info(f"WebSocket endpoint: ws://{config.host}:{config.port}/ws/chat/{{session_id}}")
    logger.info(f"Debug API: http://{config.host}:{config.port}/debug")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Chat Middleware Service - Shutting down")


# Create FastAPI application
app = FastAPI(
    title="Chat Middleware Service",
    description="WebSocket-based chat middleware for DICOM study analysis with Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include debug routes
app.include_router(debug_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}


@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for chat sessions.

    Args:
        session_id: Session identifier. Use 'new' to create a new session.
    """
    await handle_websocket(websocket, session_id)


if __name__ == "__main__":
    import uvicorn

    config = init_config()

    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info"
    )
