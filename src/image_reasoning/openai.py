# Standard library imports
import os
from abc import abstractmethod
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, cast,Awaitable
from typing_extensions import Self

# External library imports
import base64
import httpx
from PIL import Image
from deprecated import deprecated
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base

# OpenAI specific imports
import openai
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

# Legacy bridge imports from src.image_reasoning.llama_index
from src.image_reasoning.bridge.pydantic import BaseModel, Field, PrivateAttr
from src.image_reasoning.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from src.image_reasoning.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from src.image_reasoning.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from src.image_reasoning.base import CallbackManager


DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""

GPT4V_MODELS = {
    "gpt-4-vision-preview": 128000,
    "gpt-4o-mini": 128000,
}


def generic_messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"

        addtional_kwargs = message.additional_kwargs
        if addtional_kwargs:
            string_message += f"\n{addtional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
    return "\n".join(string_messages)
def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def from_openai_message(openai_message: ChatCompletionMessage) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = openai_message.role
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = openai_message.content

    function_call = None  # deprecated in OpenAI v 1.1.0

    additional_kwargs: Dict[str, Any] = {}
    if openai_message.tool_calls is not None:
        tool_calls: List[ChatCompletionMessageToolCall] = openai_message.tool_calls
        additional_kwargs.update(tool_calls=tool_calls)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)

def resolve_openai_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """ "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OPENAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "OPENAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "OPENAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or openai.api_key or ""
    final_api_base = api_base or openai.base_url or DEFAULT_OPENAI_API_BASE
    final_api_version = api_version or openai.api_version or DEFAULT_OPENAI_API_VERSION

    return final_api_key, str(final_api_base), final_api_version

def to_openai_message_dict(
    message: ChatMessage, drop_none: bool = False
) -> ChatCompletionMessageParam:
    """Convert generic message to OpenAI message dict."""
    message_dict = {
        "role": message.role.value,
        "content": message.content,
    }

    # NOTE: openai messages have additional arguments:
    # - function messages have `name`
    # - assistant messages have optional `function_call`
    message_dict.update(message.additional_kwargs)

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    return message_dict  # type: ignore
def to_openai_message_dicts(
    messages: Sequence[ChatMessage], drop_none: bool = False
) -> List[ChatCompletionMessageParam]:
    """Convert generic messages to OpenAI message dicts."""
    return [
        to_openai_message_dict(message, drop_none=drop_none) for message in messages
    ]


def encode_image_from_pillow(image: Image.Image, format: str = 'PNG') -> str:
    """Encode a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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

class BaseComponent(BaseModel):
    """Base component object to capture class names."""

    class Config:
        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: "BaseComponent") -> None:
            """Add class name to schema."""
            schema["properties"]["class_name"] = {
                "title": "Class Name",
                "type": "string",
                "default": model.class_name(),
            }

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()

        # tiktoken is not pickleable
        # state["__dict__"] = self.dict()
        state["__dict__"].pop("tokenizer", None)

        # remove local functions
        keys_to_remove = []
        for key, val in state["__dict__"].items():
            if key.endswith("_fn"):
                keys_to_remove.append(key)
            if "<lambda>" in str(val):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            state["__dict__"].pop(key, None)

        # remove private attributes -- kind of dangerous
        state["__private_attribute_values__"] = {}

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Use the __dict__ and __init__ method to set state
        # so that all variable initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            super().__setstate__(state)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class MultiModalLLMMetadata(BaseModel):
    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input when generating a response."
        ),
    )
    num_output: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    num_input_files: Optional[int] = Field(
        default=DEFAULT_NUM_INPUT_FILES,
        description="Number of input files the model can take when generating a response.",
    )
    is_function_calling_model: Optional[bool] = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )

    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )


# TODO add callback functionality


class MultiModalLLM(ChainableMixin, BaseComponent):
    """Multi-Modal LLM interface."""

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""

    @abstractmethod
    def complete(
        self, prompt: str, image_documents: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, image_documents: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""

    # ===== Async Endpoints =====

    @abstractmethod
    async def acomplete(
        self, prompt: str, image_documents: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, image_documents: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """Return query component."""
        if self.metadata.is_chat_model:
            # TODO: we don't have a separate chat component
            return MultiModalCompleteComponent(multi_modal_llm=self, **kwargs)
        else:
            return MultiModalCompleteComponent(multi_modal_llm=self, **kwargs)


class BaseMultiModalComponent(QueryComponent):
    """Base LLM component."""

    multi_modal_llm: MultiModalLLM = Field(..., description="LLM")
    streaming: bool = Field(default=False, description="Streaming mode")

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        # TODO: make callbacks work with multi-modal


class MultiModalCompleteComponent(BaseMultiModalComponent):
    """Multi-modal completion component."""

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "prompt" not in input:
            raise ValueError("Prompt must be in input dict.")

        # do special check to see if prompt is a list of chat messages
        if isinstance(input["prompt"], get_args(List[ChatMessage])):
            raise NotImplementedError(
                "Chat messages not yet supported as input to multi-modal model."
            )
        else:
            input["prompt"] = validate_and_convert_stringable(input["prompt"])

        # make sure image documents are valid
        if "image_documents" in input:
            if not isinstance(input["image_documents"], list):
                raise ValueError("image_documents must be a list.")
            for doc in input["image_documents"]:
                if not isinstance(doc, ImageDocument):
                    raise ValueError(
                        "image_documents must be a list of ImageDocument objects."
                    )

        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        prompt = kwargs["prompt"]
        image_documents = kwargs.get("image_documents", [])
        if self.streaming:
            response = self.multi_modal_llm.stream_complete(prompt, image_documents)
        else:
            response = self.multi_modal_llm.complete(prompt, image_documents)
        return {"output": response}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: support only complete for now
        # non-trivial to figure how to support chat/complete/etc.
        prompt = kwargs["prompt"]
        image_documents = kwargs.get("image_documents", [])
        if self.streaming:
            response = await self.multi_modal_llm.astream_complete(
                prompt, image_documents
            )
        else:
            response = await self.multi_modal_llm.acomplete(prompt, image_documents)
        return {"output": response}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        # TODO: support only complete for now
        return InputKeys.from_keys({"prompt", "image_documents"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


class OpenAIMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from OpenAI.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: Optional[int] = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    image_detail: str = Field(
        description="The level of details for image in API calls. Can be low, high, or auto"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries.",
        gte=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        gte=0,
    )
    api_key: str = Field(default=None, description="The OpenAI API key.", exclude=True)
    api_base: str = Field(default=None, description="The base URL for OpenAI API.")
    api_version: str = Field(description="The API version for OpenAI API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    default_headers: Dict[str, str] = Field(
        default=None, description="The default headers for API requests."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _client: SyncOpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: Optional[int] = 500,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        max_retries: int = 3,
        timeout: float = 60.0,
        image_detail: str = "low",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            image_detail=image_detail,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            callback_manager=callback_manager,
            default_headers=default_headers,
            **kwargs,
        )
        self._http_client = http_client
        self._client, self._aclient = self._get_clients(**kwargs)

    def _get_clients(self, **kwargs: Any) -> Tuple[SyncOpenAI, AsyncOpenAI]:
        client = SyncOpenAI(**self._get_credential_kwargs())
        aclient = AsyncOpenAI(**self._get_credential_kwargs())
        return client, aclient

    @classmethod
    def class_name(cls) -> str:
        return "openai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_new_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
            "timeout": self.timeout,
            **kwargs,
        }

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        images: Sequence[Image.Image],
        **kwargs: Any,
    ) -> List[ChatCompletionMessageParam]:
        return to_openai_message_dicts(
            [
                generate_openai_multi_modal_chat_message(
                    prompt=prompt,
                    role=role,
                    images=images,
                    image_detail=self.image_detail,
                )
            ]
        )

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in GPT4V_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(GPT4V_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_new_tokens is not None:
            base_kwargs["max_tokens"] = self.max_new_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _get_response_token_counts(self, raw_response: Any) -> dict:
        """Get the token usage reported by the response."""
        if not isinstance(raw_response, dict):
            return {}

        usage = raw_response.get("usage", {})
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, images=images
        )
        response = self._client.chat.completions.create(
            messages=message_dict,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = self._client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        openai_message = response.choices[0].message
        message = from_openai_message(openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _stream_complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, images=images
        )

        def gen() -> CompletionResponseGen:
            text = ""

            for response in self._client.chat.completions.create(
                messages=message_dict,
                stream=True,
                **all_kwargs,
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                content_delta = delta.content or ""
                text += content_delta

                yield CompletionResponse(
                    delta=content_delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(messages)

        def gen() -> ChatResponseGen:
            content = ""
            tool_calls: List[ChoiceDeltaToolCall] = []

            is_function = False
            for response in self._client.chat.completions.create(
                messages=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta

                additional_kwargs = {}
                if is_function:
                    tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                    additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, images, **kwargs)

    def stream_complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, images, **kwargs)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    async def _acomplete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, images=images
        )
        response = await self._aclient.chat.completions.create(
            messages=message_dict,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def acomplete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, images, **kwargs)

    async def _astream_complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, images=images
        )

        async def gen() -> CompletionResponseAsyncGen:
            text = ""

            async for response in await self._aclient.chat.completions.create(
                messages=message_dict,
                stream=True,
                **all_kwargs,
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                content_delta = delta.content or ""
                text += content_delta

                yield CompletionResponse(
                    delta=content_delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = await self._aclient.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        openai_message = response.choices[0].message
        message = from_openai_message(openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        message_dicts = to_openai_message_dicts(messages)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            tool_calls: List[ChoiceDeltaToolCall] = []

            is_function = False
            async for response in await self._aclient.chat.completions.create(
                messages=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()

                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta

                additional_kwargs = {}
                if is_function:
                    tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                    additional_kwargs["tool_calls"] = tool_calls

                yield ChatResponse(
                    message=ChatMessage(
                        role=role,
                        content=content,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    async def astream_complete(
        self, prompt: str, images: Sequence[Image.Image], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, images, **kwargs)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)
