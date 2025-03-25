from collections.abc import Callable
from typing import Protocol, get_type_hints

from aikernel import Conversation, LLMTool
from pydantic import BaseModel


class IToolFn[ContextT, ParametersT: BaseModel, ReturnT](Protocol):
    __name__: str

    async def __call__(self, *, context: ContextT, parameters: ParametersT, conversation: Conversation) -> ReturnT: ...


class Tool[ContextT, ParametersT: BaseModel, ReturnT: BaseModel]:
    def __init__(self, fn: IToolFn[ContextT, ParametersT, ReturnT], /, *, name: str | None = None) -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name or self._fn.__name__

    @property
    def description(self) -> str:
        return self._fn.__doc__ or ""

    @property
    def parameters_model(self) -> type[ParametersT]:
        type_hints = get_type_hints(self._fn)
        parameters_type_hint = type_hints.get("parameters")

        if parameters_type_hint is not None:
            return parameters_type_hint
        else:
            raise TypeError(
                "Invalid type signature for Tool; `use` method must have a single `parameters` parameter with a Pydantic model type"
            )

    def as_llm_tool(self) -> LLMTool[ParametersT]:
        return LLMTool(name=self.name, description=self.description, parameters=self.parameters_model)

    async def __call__(self, *, context: ContextT, parameters: ParametersT, conversation: Conversation) -> ReturnT:
        return await self._fn(context=context, parameters=parameters, conversation=conversation)


def tool[ContextT, ParametersT: BaseModel, ReturnT: BaseModel](
    *, name: str
) -> Callable[[IToolFn[ContextT, ParametersT, ReturnT]], Tool[ContextT, ParametersT, ReturnT]]:
    def decorator(fn: IToolFn[ContextT, ParametersT, ReturnT]) -> Tool[ContextT, ParametersT, ReturnT]:
        return Tool(fn, name=name)

    return decorator
