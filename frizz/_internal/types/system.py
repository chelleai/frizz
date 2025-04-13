from typing import Any, Protocol

from aikernel import LLMMessagePart

from frizz._internal.tools import Tool


class IGetToolSystemMessagePart(Protocol):
    """Protocol for generating system message parts that describe available tools.
    
    This protocol defines a callable that generates a message part containing
    information about the tools available to the agent. This message part is used
    to enhance the system message when the agent needs to be aware of available tools.
    """
    def __call__(self, *, tools: list[Tool[Any, Any, Any]]) -> LLMMessagePart: 
        """Generate a message part describing available tools.
        
        Args:
            tools: A list of tools available to the agent.
            
        Returns:
            A message part describing the available tools for use in the system message.
        """
        ...
