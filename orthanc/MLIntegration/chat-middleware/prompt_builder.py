"""
Prompt builder for assembling OpenAI-compatible message arrays
with interleaved image_url and text content parts.
"""
import logging
from typing import Dict, List, Tuple, Union

from runtime_config import RuntimeConfig, get_runtime_config

logger = logging.getLogger(__name__)

ContentType = Union[str, List[dict]]


class PromptBuilder:
    """
    Builds OpenAI-compatible message arrays from conversation history
    and current request with images.

    Produces interleaved content arrays with explicit slice labels,
    matching the format expected by MedGemma via Ollama's
    /v1/chat/completions endpoint.
    """

    def __init__(self, runtime_config: RuntimeConfig = None):
        self._config = runtime_config

    @property
    def config(self) -> RuntimeConfig:
        if self._config is None:
            return get_runtime_config()
        return self._config

    def _build_user_content(
        self,
        user_text: str,
        series_images: Dict[str, List[str]]
    ) -> ContentType:
        """
        Build the content value for a user message.

        With images: returns an interleaved content array:
            [image_url, "SLICE 1", image_url, "SLICE 2", ..., user_text]
        Without images: returns the plain text string.
        """
        all_images = []
        for series_uid, images in series_images.items():
            all_images.extend(images)
            logger.debug(f"Added {len(images)} images from series {series_uid}")

        if not all_images:
            logger.info("Built user content with no images (text-only)")
            return user_text

        content: List[dict] = []
        for idx, data_uri in enumerate(all_images, start=1):
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
            content.append({"type": "text", "text": f"SLICE {idx}"})

        content.append({"type": "text", "text": user_text})

        logger.info(f"Built user content with {len(all_images)} images (interleaved)")
        return content

    def build_messages(
        self,
        conversation_history: List[dict],
        new_user_message: str,
        series_images: Dict[str, List[str]]
    ) -> Tuple[List[dict], ContentType]:
        """
        Build OpenAI-format messages array for /v1/chat/completions.

        Returns:
            Tuple of (messages_for_api, user_content_to_store_in_history)
        """
        messages = []

        # 1. System prompt
        messages.append({
            "role": "system",
            "content": self.config.system_prompt
        })

        # 2. Replay conversation history as-is
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 3. Current user message (interleaved content array or plain text)
        user_content = self._build_user_content(new_user_message, series_images)
        messages.append({"role": "user", "content": user_content})

        logger.debug(
            f"Total messages in prompt: {len(messages)} "
            f"(1 system + {len(conversation_history)} history + 1 current)"
        )

        return messages, user_content


# Global prompt builder instance
_prompt_builder: PromptBuilder = None


def get_prompt_builder() -> PromptBuilder:
    """Get the global prompt builder singleton"""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
