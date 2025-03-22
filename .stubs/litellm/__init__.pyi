from typing import Any, Literal, NotRequired, TypedDict

_LiteLLMGeminiModel = Literal[
    "vertex_ai/gemini-1.5-flash",
    "vertex_ai/gemini-1.5-pro",
    "vertex_ai/gemini-1.5-flash-8b",
    "vertex_ai/gemini-2.0-flash",
]
_LiteLLMEmbeddingModel = Literal[
    "gemini/text-embedding-004",
]
_MessageRole = Literal["system", "user", "assistant"]


# request
class _LiteLLMTextMessageContent(TypedDict):
    type: Literal["text"]
    text: str


class _LiteLLMMediaMessageContent(TypedDict):
    type: Literal["image_url"]
    image_url: str


class _LiteLLMCacheControl(TypedDict):
    type: Literal["ephemeral"]


class _LiteLLMMessage(TypedDict):
    role: _MessageRole
    content: list[_LiteLLMTextMessageContent | _LiteLLMMediaMessageContent]
    cache_control: NotRequired[_LiteLLMCacheControl]


class _LiteLLMFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _LiteLLMTool(TypedDict):
    type: Literal["function"]
    function: _LiteLLMFunction


# response
class _LiteLLMModelResponseChoiceToolCallFunction:
    name: str
    arguments: str


class _LiteLLMModelResponseChoiceToolCall:
    function: _LiteLLMModelResponseChoiceToolCallFunction


class _LiteLLMModelResponseChoiceMessage:
    role: Literal["assistant"]
    content: str
    tool_calls: list[_LiteLLMModelResponseChoiceToolCall]

class _LiteLLMModelResponseChoice:
    finish_reason: Literal["stop"]
    index: int
    message: _LiteLLMModelResponseChoiceMessage


class _LiteLLMUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ModelResponse:
    id: str
    created: int
    model: _LiteLLMGeminiModel
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[_LiteLLMModelResponseChoice]
    usage: _LiteLLMUsage


class _LiteLLMEmbeddingData(TypedDict):
    index: int
    object: Literal["embedding"]
    embedding: list[float]


class EmbeddingResponse:
    data: list[_LiteLLMEmbeddingData]
    model: _LiteLLMEmbeddingModel
    usage: _LiteLLMUsage


async def acompletion(
    *,
    model: _LiteLLMGeminiModel,
    messages: list[_LiteLLMMessage],
    response_format: Any = None,
    tools: list[_LiteLLMTool] | None = None,
    tool_choice: Literal["auto", "none", "required"] | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
) -> ModelResponse: ...


async def aembedding(model: _LiteLLMEmbeddingModel, input: list[str]) -> EmbeddingResponse: ...