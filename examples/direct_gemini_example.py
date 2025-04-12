"""
Direct Gemini Example using Vertex AI SDK

This example demonstrates how to use the Frizz package with Google's Vertex AI SDK
to directly interact with the Gemini model for tool calling.
"""
import asyncio
import json
import os
from typing import Any, Literal

# Import Vertex AI SDK
from google.cloud import aiplatform
from pydantic import BaseModel
from vertexai.preview.generative_models import FunctionDeclaration, GenerativeModel, Tool, Type

# Import Frizz components - we only need the core functionality
from frizz import Tool as FrizzTool
from frizz import tool


# Define Parameter and Result models for calculator
class CalculatorParams(BaseModel):
    """Parameters for the calculator tool."""
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorResult(BaseModel):
    """Return type for the calculator tool."""
    result: float
    operation: str


# Define a context class
class MyContext:
    """Simple context for demonstration."""
    pass


# Create a calculator tool
@tool(name="calculator")
async def calculator(*, context: MyContext, parameters: CalculatorParams, conversation=None) -> CalculatorResult:
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


class DirectGeminiAgent:
    """A simplified agent that directly uses Vertex AI and the Gemini model."""
    
    def __init__(self, tools: list[FrizzTool], context: Any, api_endpoint: str | None = None):
        """Initialize the agent with tools and context."""
        self.tools = tools
        self.context = context
        self.conversation_history = []
        
        # Initialize Vertex AI
        aiplatform.init(location=os.getenv("VERTEX_LOCATION", "us-central1"))
        
        # Convert Frizz tools to Vertex AI tools
        self.vertex_tools = []
        for tool_item in tools:
            # Get the parameter schema from the Pydantic model
            param_schema = tool_item.parameters_model.model_json_schema()
            
            # Convert to Vertex AI tool format
            vertex_tool = Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=tool_item.name,
                        description=tool_item.description,
                        parameters=Type.from_dict(param_schema)
                    )
                ]
            )
            self.vertex_tools.append(vertex_tool)
        
        # Create the model
        self.model = GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=self.vertex_tools
        )
        
    async def send_message(self, message: str) -> dict[str, Any]:
        """Send a message to the model and process any tool calls."""
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Create a chat session
        chat = self.model.start_chat(history=[
            {"role": entry["role"], "parts": [entry["content"]]} 
            for entry in self.conversation_history
        ])
        
        # Get response from model
        response = chat.send_message(message)
        
        # Check if there's a tool call
        tool_calls = []
        tool_results = []
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    # We have a tool call
                    function_call = part.function_call
                    tool_name = function_call.name
                    tool_args = json.loads(function_call.args)
                    
                    # Find the matching tool
                    matching_tool = next((t for t in self.tools if t.name == tool_name), None)
                    if matching_tool:
                        # Convert args to parameters model
                        parameters = matching_tool.parameters_model.model_validate(tool_args)
                        
                        # Execute the tool
                        result = await matching_tool(
                            context=self.context,
                            parameters=parameters,
                            conversation=None  # No conversation needed for this example
                        )
                        
                        # Add the tool call and result to our tracking
                        tool_calls.append({"name": tool_name, "args": tool_args})
                        tool_results.append({"name": tool_name, "result": result.model_dump()})
        
        # Extract the response text
        response_text = response.text
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # If there were tool calls, add their results to the history
        if tool_results:
            self.conversation_history.append({"role": "tool", "content": json.dumps(tool_results)})
        
        return {
            "response": response_text,
            "tool_calls": tool_calls,
            "tool_results": tool_results
        }


async def main():
    # Create context
    context = MyContext()
    
    # Create agent with calculator tool
    agent = DirectGeminiAgent(tools=[calculator], context=context)
    
    # Example conversation
    print("Starting conversation with the calculator assistant...\n")
    
    # First message
    user_message = "What is 125 * 37?"
    print(f"User: {user_message}")
    
    result = await agent.send_message(user_message)
    print(f"Assistant: {result['response']}")
    
    if result["tool_calls"]:
        for i, tool_call in enumerate(result["tool_calls"]):
            print(f"\nTool Call {i+1}: {tool_call['name']}({tool_call['args']})")
            print(f"Tool Result: {result['tool_results'][i]['result']}")
    
    # Second message
    user_message = "If I have 250 items that cost $13.50 each, what's my total cost?"
    print(f"\nUser: {user_message}")
    
    result = await agent.send_message(user_message)
    print(f"Assistant: {result['response']}")
    
    if result["tool_calls"]:
        for i, tool_call in enumerate(result["tool_calls"]):
            print(f"\nTool Call {i+1}: {tool_call['name']}({tool_call['args']})")
            print(f"Tool Result: {result['tool_results'][i]['result']}")


if __name__ == "__main__":
    asyncio.run(main())
