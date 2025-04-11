from collections.abc import Callable
from typing import Protocol, get_type_hints

from aikernel import Conversation, LLMTool
from pydantic import BaseModel


class IToolFn[ContextT, ParametersT: BaseModel, ReturnT](Protocol):
    """Protocol defining the interface for tool functions.
    
    A tool function is a callable that processes a request with a given context and parameters,
    and returns a result. It provides functionality that can be used by an agent during a conversation.
    
    Type Parameters:
        ContextT: The type of the context object that will be passed to the tool function.
        ParametersT: The Pydantic model type that defines the parameters for the tool.
        ReturnT: The return type of the tool function.
    """
    __name__: str

    async def __call__(self, *, context: ContextT, parameters: ParametersT, conversation: Conversation) -> ReturnT: ...


class Tool[ContextT, ParametersT: BaseModel, ReturnT: BaseModel]:
    """Represents a tool that can be used by an agent during a conversation.
    
    A tool wraps a function that provides some functionality to the agent. The tool includes
    metadata such as a name and description, and handles parameter validation and execution.
    
    Type Parameters:
        ContextT: The type of the context object that will be passed to the tool function.
        ParametersT: The Pydantic model type that defines the parameters for the tool.
        ReturnT: The return type of the tool function, must be a Pydantic model.
    """
    def __init__(self, fn: IToolFn[ContextT, ParametersT, ReturnT], /, *, name: str | None = None) -> None:
        """Initialize a Tool instance.
        
        Args:
            fn: The function implementing the tool's functionality.
            name: Optional name for the tool. If not provided, the function name will be used.
        """
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the tool.
        
        Returns:
            The name of the tool, either as provided during initialization or the function name.
        """
        return self._name or self._fn.__name__

    @property
    def description(self) -> str:
        """Get the description of the tool.
        
        Returns:
            The docstring of the function, or an empty string if no docstring is present.
        """
        return self._fn.__doc__ or ""

    @property
    def parameters_model(self) -> type[ParametersT]:
        """Get the Pydantic model type for the tool's parameters.
        
        Returns:
            The type of the parameters model for this tool.
            
        Raises:
            TypeError: If the tool function doesn't have a properly typed parameters parameter.
        """
        type_hints = get_type_hints(self._fn)
        parameters_type_hint = type_hints.get("parameters")

        if parameters_type_hint is not None:
            return parameters_type_hint
        else:
            raise TypeError(
                "Invalid type signature for Tool; `use` method must have a single `parameters` parameter with a Pydantic model type"
            )

    def as_llm_tool(self) -> LLMTool[ParametersT]:
        """Convert this tool to an LLMTool for use with the aikernel library.
        
        Returns:
            An LLMTool instance representing this tool.
        """
        return LLMTool(name=self.name, description=self.description, parameters=self.parameters_model)

    async def __call__(self, *, context: ContextT, parameters: ParametersT, conversation: Conversation) -> ReturnT:
        """Call the tool function with the given context and parameters.
        
        Args:
            context: The context object to pass to the tool function.
            parameters: The validated parameters for the tool function.
            conversation: The current conversation.
            
        Returns:
            The result of calling the tool function.
        """
        return await self._fn(context=context, parameters=parameters, conversation=conversation)


def tool[ContextT, ParametersT: BaseModel, ReturnT: BaseModel](
    *, name: str
) -> Callable[[IToolFn[ContextT, ParametersT, ReturnT]], Tool[ContextT, ParametersT, ReturnT]]:
    """Decorator for creating a Tool from a function.
    
    This decorator allows for easy creation of Tool instances by decorating functions.
    
    Type Parameters:
        ContextT: The type of the context object that will be passed to the tool function.
        ParametersT: The Pydantic model type that defines the parameters for the tool.
        ReturnT: The return type of the tool function, must be a Pydantic model.
        
    Args:
        name: The name to use for the tool.
        
    Returns:
        A decorator function that wraps a tool function and returns a Tool instance.
    
    Example:
        ```python
        @tool(name="get_weather")
        async def get_weather(*, context: MyContext, parameters: WeatherParams, conversation: Conversation) -> WeatherResult:
            '''Get the current weather for a location.'''
            # Implementation...
            return WeatherResult(...)
        ```
    """
    def decorator(fn: IToolFn[ContextT, ParametersT, ReturnT]) -> Tool[ContextT, ParametersT, ReturnT]:
        return Tool(fn, name=name)

    return decorator
