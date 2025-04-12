"""
Example 1: Creating a Basic AI Assistant That Can Call Functions

This example demonstrates how to create a simple assistant that can use a
calculator tool to perform arithmetic operations during a conversation.
"""
import asyncio

from aikernel import Conversation, LLMMessagePart, LLMRouter, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import BaseModel

from frizz import Agent, tool


# Define parameter and return models for our calculator tool
class CalculatorParams(BaseModel):
    """Parameters for the calculator tool."""
    operation: str  # add, subtract, multiply, divide
    a: float
    b: float


class CalculatorResult(BaseModel):
    """Return type for the calculator tool."""
    result: float
    operation: str


# Define a simple context class that could contain configuration or dependencies
class MyContext:
    """Example context with no specific requirements."""
    pass


# Create a calculator tool that the assistant can use
@tool(name="calculator")
async def calculator(*, context: MyContext, parameters: CalculatorParams, conversation: Conversation) -> CalculatorResult:
    """Perform basic arithmetic operations.
    
    Supported operations: add, subtract, multiply, divide.
    """
    operation = parameters.operation.lower()
    a = parameters.a
    b = parameters.b
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return CalculatorResult(result=result, operation=operation)


async def main():
    # Create an agent with our calculator tool
    agent = Agent(
        tools=[calculator],
        context=MyContext(),
        system_message=LLMSystemMessage(parts=[LLMMessagePart(content="""
            You are a helpful assistant that can perform calculations.
            When asked to perform arithmetic, use the calculator tool rather than calculating yourself.
        """)])
    )
    
    # Create a router for the LLM API
    router = get_router(models=("claude-3-sonnet-20240229",))
    
    # Print a note about the expected validation errors
    print("Note: You may see validation errors related to message content being None.")
    print("This is expected behavior when the model makes a tool call and doesn't provide text content.\n")
    
    # Example conversation
    print("Starting conversation with the calculator assistant...\n")
    
    # First user message asking for a calculation
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="What is 125 * 37?")])
    print(f"User: {user_message.content}")
    
    # Let the agent process the message
    # Pass both router and model to step method
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    if result.tool_message:
        print(f"Tool Result: {result.tool_message}")
    
    # Second user message with a more complex request
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="If I have 250 items that cost $13.50 each, what's my total cost?")])
    print(f"\nUser: {user_message.content}")
    
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")


if __name__ == "__main__":
    asyncio.run(main())
