from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, get_args

from src.image_reasoning.llama_index.legacy.bridge.pydantic import BaseModel, Field
from src.image_reasoning.llama_index.legacy.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_INPUT_FILES,
    DEFAULT_NUM_OUTPUTS,
)
from src.image_reasoning.llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from src.image_reasoning.llama_index.legacy.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from src.image_reasoning.llama_index.legacy.schema import BaseComponent, ImageDocument



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
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
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
