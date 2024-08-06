import logging
from typing import Any, Dict, Optional, Sequence
from PIL import Image

from src.image_reasoning.llama_index.legacy.multi_model_llms.base import ChatMessage
from src.image_reasoning.llama_index.legacy.multi_model_llms.generic_utils import encode_image, encode_image_from_pillow
from src.image_reasoning.llama_index.legacy.schema import ImageDocument


DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


GPT4V_MODELS = {
    "gpt-4-vision-preview": 128000,
    "gpt-4o-mini": 128000,
}


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""

logger = logging.getLogger(__name__)


# def generate_openai_multi_modal_chat_message(
#     prompt: str,
#     role: str,
#     image_documents: Optional[Sequence[ImageDocument]] = None,
#     image_detail: Optional[str] = "low",
# ) -> ChatMessage:
#     # if image_documents is empty, return text only chat message
#     if image_documents is None:
#         return ChatMessage(role=role, content=prompt)

#     # if image_documents is not empty, return text with images chat message
#     completion_content = [{"type": "text", "text": prompt}]
#     for image_document in image_documents:
#         image_content: Dict[str, Any] = {}
#         mimetype = image_document.image_mimetype or "image/png"
#         if isinstance(image_document.image, str):
#             base64_image = image_document.image
#         elif isinstance(image_document.image, Image.Image):
#             base64_image = encode_image_from_pillow(image_document.image, format='PNG')
#         elif image_document.image_path and image_document.image_path != "":
#             base64_image = encode_image(image_document.image_path)
#         else:
#             raise ValueError("No valid image found in ImageDocument.")

#         image_content = {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:{mimetype};base64,{base64_image}",
#                 "detail": image_detail,
#             },
#         }
#         completion_content.append(image_content)

#     return ChatMessage(role=role, content=completion_content)

def generate_openai_multi_modal_chat_message(
    prompt: str,
    role: str,
    images: Optional[Sequence[Image.Image]] = None,
    image_detail: Optional[str] = "low",
) -> ChatMessage:
    # if images is empty, return text only chat message
    if images is None:
        return ChatMessage(role=role, content=prompt)

    # if images is not empty, return text with images chat message
    completion_content = [{"type": "text", "text": prompt}]
    for image in images:
        image_content: Dict[str, Any] = {}
        base64_image = encode_image_from_pillow(image, format='PNG')
        mimetype = "image/png"

        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mimetype};base64,{base64_image}",
                "detail": image_detail,
            },
        }
        completion_content.append(image_content)

    return ChatMessage(role=role, content=completion_content)
