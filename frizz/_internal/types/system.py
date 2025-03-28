from typing import Any, Protocol

from aikernel import LLMMessagePart

from frizz._internal.tools import Tool


class IGetToolSystemMessagePart(Protocol):
    def __call__(self, *, tools: list[Tool[Any, Any, Any]]) -> LLMMessagePart: ...
