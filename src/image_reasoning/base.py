import logging
import uuid
from abc import ABC,abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# timestamp for callback events
TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S.%f"

# base trace_id for the tracemap in callback_manager
BASE_TRACE_EVENT = "root"

logger = logging.getLogger(__name__)
global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])
empty_trace_ids: List[str] = []
global_stack_trace_ids = ContextVar("trace_ids", default=empty_trace_ids)





class CBEventType(str, Enum):
    """Callback manager event types.

    Attributes:
        CHUNKING: Logs for the before and after of text splitting.
        NODE_PARSING: Logs for the documents and the nodes that they are parsed into.
        EMBEDDING: Logs for the number of texts embedded.
        LLM: Logs for the template and response of LLM calls.
        QUERY: Keeps track of the start and end of each query.
        RETRIEVE: Logs for the nodes retrieved for a query.
        SYNTHESIZE: Logs for the result for synthesize calls.
        TREE: Logs for the summary and level of summaries generated.
        SUB_QUESTION: Logs for a generated sub question and answer.
    """

    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"


class EventPayload(str, Enum):
    DOCUMENTS = "documents"  # list of documents before parsing
    CHUNKS = "chunks"  # list of text chunks
    NODES = "nodes"  # list of nodes
    PROMPT = "formatted_prompt"  # formatted prompt sent to LLM
    MESSAGES = "messages"  # list of messages sent to LLM
    COMPLETION = "completion"  # completion from LLM
    RESPONSE = "response"  # message response from LLM
    QUERY_STR = "query_str"  # query used for query engine
    SUB_QUESTION = "sub_question"  # a sub question & answer + sources
    EMBEDDINGS = "embeddings"  # list of embeddings
    TOP_K = "top_k"  # top k nodes retrieved
    ADDITIONAL_KWARGS = "additional_kwargs"  # additional kwargs for event call
    SERIALIZED = "serialized"  # serialized object for event caller
    FUNCTION_CALL = "function_call"  # function call for the LLM
    FUNCTION_OUTPUT = "function_call_response"  # function call output
    TOOL = "tool"  # tool used in LLM call
    MODEL_NAME = "model_name"  # model name used in an event
    TEMPLATE = "template"  # template used in LLM call
    TEMPLATE_VARS = "template_vars"  # template variables used in LLM call
    SYSTEM_PROMPT = "system_prompt"  # system prompt used in LLM call
    QUERY_WRAPPER_PROMPT = "query_wrapper_prompt"  # query wrapper prompt used in LLM
    EXCEPTION = "exception"  # exception raised in an event


# events that will never have children events
LEAF_EVENTS = (CBEventType.CHUNKING, CBEventType.LLM, CBEventType.EMBEDDING)


@dataclass
class CBEvent:
    """Generic class to store event information."""

    event_type: CBEventType
    payload: Optional[Dict[str, Any]] = None
    time: str = ""
    id_: str = ""

    def __post_init__(self) -> None:
        """Init time and id if needed."""
        if not self.time:
            self.time = datetime.now().strftime(TIMESTAMP_FORMAT)
        if not self.id_:
            self.id = str(uuid.uuid4())


@dataclass
class EventStats:
    """Time-based Statistics for events."""

    total_secs: float
    average_secs: float
    total_count: int


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to track event starts and ends."""

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType],
        event_ends_to_ignore: List[CBEventType],
    ) -> None:
        """Initialize the base callback handler."""
        self.event_starts_to_ignore = tuple(event_starts_to_ignore)
        self.event_ends_to_ignore = tuple(event_ends_to_ignore)

    @abstractmethod
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""

    @abstractmethod
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""

    @abstractmethod
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

    @abstractmethod
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""



class CallbackManager(BaseCallbackHandler, ABC):
    """
    Callback manager that handles callbacks for events within LlamaIndex.

    The callback manager provides a way to call handlers on event starts/ends.

    Additionally, the callback manager traces the current stack of events.
    It does this by using a few key attributes.
    - trace_stack - The current stack of events that have not ended yet.
                    When an event ends, it's removed from the stack.
                    Since this is a contextvar, it is unique to each
                    thread/task.
    - trace_map - A mapping of event ids to their children events.
                  On the start of events, the bottom of the trace stack
                  is used as the current parent event for the trace map.
    - trace_id - A simple name for the current trace, usually denoting the
                 entrypoint (query, index_construction, insert, etc.)

    Args:
        handlers (List[BaseCallbackHandler]): list of handlers to use.

    Usage:
        with callback_manager.event(CBEventType.QUERY) as event:
            event.on_start(payload={key, val})
            ...
            event.on_end(payload={key, val})

    """

    def __init__(self, handlers: Optional[List[BaseCallbackHandler]] = None):
        """Initialize the manager with a list of handlers."""
        from llama_index.legacy import global_handler

        handlers = handlers or []

        # add eval handlers based on global defaults
        if global_handler is not None:
            new_handler = global_handler
            # go through existing handlers, check if any are same type as new handler
            # if so, error
            for existing_handler in handlers:
                if isinstance(existing_handler, type(new_handler)):
                    raise ValueError(
                        "Cannot add two handlers of the same type "
                        f"{type(new_handler)} to the callback manager."
                    )
            handlers.append(new_handler)

        self.handlers = handlers
        self._trace_map: Dict[str, List[str]] = defaultdict(list)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run handlers when an event starts and return id of event."""
        event_id = event_id or str(uuid.uuid4())

        # if no trace is running, start a default trace
        try:
            parent_id = parent_id or global_stack_trace.get()[-1]
        except IndexError:
            self.start_trace("llama-index")
            parent_id = global_stack_trace.get()[-1]

        self._trace_map[parent_id].append(event_id)
        for handler in self.handlers:
            if event_type not in handler.event_starts_to_ignore:
                handler.on_event_start(
                    event_type,
                    payload,
                    event_id=event_id,
                    parent_id=parent_id,
                    **kwargs,
                )

        if event_type not in LEAF_EVENTS:
            # copy the stack trace to prevent conflicts with threads/coroutines
            current_trace_stack = global_stack_trace.get().copy()
            current_trace_stack.append(event_id)
            global_stack_trace.set(current_trace_stack)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event ends."""
        event_id = event_id or str(uuid.uuid4())
        for handler in self.handlers:
            if event_type not in handler.event_ends_to_ignore:
                handler.on_event_end(event_type, payload, event_id=event_id, **kwargs)

        if event_type not in LEAF_EVENTS:
            # copy the stack trace to prevent conflicts with threads/coroutines
            current_trace_stack = global_stack_trace.get().copy()
            current_trace_stack.pop()
            global_stack_trace.set(current_trace_stack)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers

    @contextmanager
    def event(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> Generator["EventContext", None, None]:
        """Context manager for lanching and shutdown of events.

        Handles sending on_evnt_start and on_event_end to handlers for specified event.

        Usage:
            with callback_manager.event(CBEventType.QUERY, payload={key, val}) as event:
                ...
                event.on_end(payload={key, val})  # optional
        """
        # create event context wrapper
        event = EventContext(self, event_type, event_id=event_id)
        event.on_start(payload=payload)

        payload = None
        try:
            yield event
        except Exception as e:
            # data already logged to trace?
            if not hasattr(e, "event_added"):
                payload = {EventPayload.EXCEPTION: e}
                e.event_added = True  # type: ignore
                if not event.finished:
                    event.on_end(payload=payload)
            raise
        finally:
            # ensure event is ended
            if not event.finished:
                event.on_end(payload=payload)

    @contextmanager
    def as_trace(self, trace_id: str) -> Generator[None, None, None]:
        """Context manager tracer for lanching and shutdown of traces."""
        self.start_trace(trace_id=trace_id)

        try:
            yield
        except Exception as e:
            # event already added to trace?
            if not hasattr(e, "event_added"):
                self.on_event_start(
                    CBEventType.EXCEPTION, payload={EventPayload.EXCEPTION: e}
                )
                e.event_added = True  # type: ignore

            raise
        finally:
            # ensure trace is ended
            self.end_trace(trace_id=trace_id)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        current_trace_stack_ids = global_stack_trace_ids.get().copy()
        if trace_id is not None:
            if len(current_trace_stack_ids) == 0:
                self._reset_trace_events()

                for handler in self.handlers:
                    handler.start_trace(trace_id=trace_id)

                current_trace_stack_ids = [trace_id]
            else:
                current_trace_stack_ids.append(trace_id)

        global_stack_trace_ids.set(current_trace_stack_ids)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        current_trace_stack_ids = global_stack_trace_ids.get().copy()
        if trace_id is not None and len(current_trace_stack_ids) > 0:
            current_trace_stack_ids.pop()
            if len(current_trace_stack_ids) == 0:
                for handler in self.handlers:
                    handler.end_trace(trace_id=trace_id, trace_map=self._trace_map)
                current_trace_stack_ids = []

        global_stack_trace_ids.set(current_trace_stack_ids)

    def _reset_trace_events(self) -> None:
        """Helper function to reset the current trace."""
        self._trace_map = defaultdict(list)
        global_stack_trace.set([BASE_TRACE_EVENT])

    @property
    def trace_map(self) -> Dict[str, List[str]]:
        return self._trace_map


class EventContext:
    """
    Simple wrapper to call callbacks on event starts and ends
    with an event type and id.
    """

    def __init__(
        self,
        callback_manager: CallbackManager,
        event_type: CBEventType,
        event_id: Optional[str] = None,
    ):
        self._callback_manager = callback_manager
        self._event_type = event_type
        self._event_id = event_id or str(uuid.uuid4())
        self.started = False
        self.finished = False

    def on_start(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not self.started:
            self.started = True
            self._callback_manager.on_event_start(
                self._event_type, payload=payload, event_id=self._event_id, **kwargs
            )
        else:
            logger.warning(
                f"Event {self._event_type!s}: {self._event_id} already started!"
            )

    def on_end(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not self.finished:
            self.finished = True
            self._callback_manager.on_event_end(
                self._event_type, payload=payload, event_id=self._event_id, **kwargs
            )
