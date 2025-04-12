"""
Example demonstrating how to use the Agent with a patched step method.

This example modifies the Agent class's step method to work with the current
version of the aikernel package by removing the model parameter.
"""
import asyncio
from typing import Any

from aikernel import (
    Conversation,
    LLMMessagePart,
    LLMRouter,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_structured,
    llm_tool_call,
)
from pydantic import BaseModel

from frizz import Tool
from frizz._internal.types.response import AgentMessage, StepResult
from frizz.errors import FrizzError


# Patched version of the Agent class that works with the current aikernel version
class PatchedAgent:
    def __init__(
        self,
        *,
        tools: list[Tool[Any, Any, Any]],
        context: Any,
        system_message: LLMSystemMessage | None = None,
        conversation_dump: str | None = None,
        get_tools_system_message_part: Any | None = None,
    ) -> None:
        """Initialize a new Agent instance."""
        self._tools = tools
        self._context = context
        self._conversation = (
            Conversation.load(dump=conversation_dump) if conversation_dump is not None else Conversation()
        )
        if system_message is not None:
            self._conversation.set_system_message(message=system_message)

        self._get_tools_system_message_part = get_tools_system_message_part

    @property
    def conversation(self) -> Conversation:
        """Get the current conversation."""
        return self._conversation

    @property
    def tools_by_name(self) -> dict[str, Tool[Any, BaseModel, BaseModel]]:
        """Get a dictionary of tools indexed by name."""
        return {tool.name: tool for tool in self._tools}
    
    async def step(self, *, user_message: LLMUserMessage, router: LLMRouter) -> StepResult:
        """Process a single step of conversation with the agent.
        
        Key change: Removed the model parameter as it's not used in the llm_structured function.
        """
        with self.conversation.session():
            self._conversation.add_user_message(message=user_message)

            # Use llm_structured without the model parameter
            agent_message = await llm_structured(
                messages=self._conversation.render(),
                response_model=AgentMessage,
                router=router,
            )

            assistant_message = LLMSystemMessage(
                parts=[LLMMessagePart(content=agent_message.text)]
            )
            self._conversation.add_assistant_message(message=assistant_message)

            if agent_message.chosen_tool_name is not None:
                chosen_tool = self.tools_by_name.get(agent_message.chosen_tool_name)
                if chosen_tool is None:
                    raise FrizzError(f"Tool {agent_message.chosen_tool_name} not found")

                try:
                    parameters_response = await llm_tool_call(
                        messages=self._conversation.render(),
                        tools=[chosen_tool.as_llm_tool()],
                        tool_choice="required",
                        router=router,
                    )
                    parameters = chosen_tool.parameters_model.model_validate(parameters_response.tool_call.arguments)
                except Exception as error:
                    raise FrizzError(f"Error with tool parameters: {error}")

                try:
                    await chosen_tool(
                        context=self._context, parameters=parameters, conversation=self._conversation
                    )
                except Exception as error:
                    raise FrizzError(f"Error calling tool: {error}")

                # Here, tool_message creation is simplified
                tool_message = None  # We'd need the full implementation to create a proper tool message
            else:
                tool_message = None

        return StepResult(assistant_message=assistant_message, tool_message=tool_message)


# Define simple parameter and return models
class EchoParams(BaseModel):
    """Parameters for the echo tool."""
    message: str


class EchoResult(BaseModel):
    """Return type for the echo tool."""
    message: str


# Simple context class
class SimpleContext:
    """Minimal context."""
    pass


# Create a simple tool function
async def echo_tool(*, context: SimpleContext, parameters: EchoParams, conversation: Conversation) -> EchoResult:
    """Echo back the input message."""
    return EchoResult(message=f"Echo: {parameters.message}")


# Create a Tool instance manually
echo_instance = Tool(echo_tool, name="echo")


async def main():
    # Create an agent with our tool
    agent = PatchedAgent(
        tools=[echo_instance],
        context=SimpleContext(),
        system_message=LLMSystemMessage(
            parts=[LLMMessagePart(content="You are a helpful assistant.")]
        )
    )
    
    # Create a router
    router = get_router(models=("claude-3.7-sonnet",))
    
    # Simple conversation
    print("Starting conversation with the patched agent...\n")
    
    # Create a user message
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Hello, world!")])
    print(f"User: {user_message.parts[0].content}")
    
    try:
        # Step with the agent (note: no model parameter here)
        result = await agent.step(
            user_message=user_message,
            router=router
        )
        
        print(f"Assistant: {result.assistant_message.parts[0].content}")
        if result.tool_message:
            print(f"Tool result: {result.tool_message}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
