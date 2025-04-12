"""
Simplified test example to verify the correct API usage.
"""
import asyncio

from aikernel import Conversation, LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import BaseModel

from frizz import Agent, tool


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


# Create a simple tool
@tool(name="echo")
async def echo(*, context: SimpleContext, parameters: EchoParams, conversation: Conversation) -> EchoResult:
    """Echo back the input message."""
    return EchoResult(message=f"Echo: {parameters.message}")


async def main():
    # Create an agent with our tool
    agent = Agent(
        tools=[echo],
        context=SimpleContext(),
        system_message=LLMSystemMessage(
            parts=[LLMMessagePart(content="You are a helpful assistant.")]
        )
    )
    
    # Create a router
    router = get_router(models=("claude-3.7-sonnet",))
    
    # Simple conversation
    print("Starting conversation with the echo assistant...\n")
    
    # Create a user message
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Hello, world!")])
    print(f"User: {user_message.parts[0].content}")
    
    try:
        # Step with the agent
        result = await agent.step(
            user_message=user_message,
            model="claude-3.7-sonnet",
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