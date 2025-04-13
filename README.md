# Frizz

A tool-assisted AI conversationalist framework for Python.

## Overview

Frizz provides a lightweight framework for creating AI agents that can use tools to assist in conversations. It enables:

1. Creating AI assistants that can call functions during conversations
2. Defining custom tools with typed parameters and validation
3. Managing conversation state and context across interactions
4. Structured communication between LLMs and external systems
5. Tool-assisted responses where the AI decides when to use tools

### 1. Creating AI assistants that can call functions during conversations

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant Tool

    User->>Agent: Send message
    Agent->>LLM: Forward message with context
    LLM-->>Agent: Decision to use tool
    Agent->>Tool: Call function with parameters
    Tool-->>Agent: Return result
    Agent->>LLM: Include tool result
    LLM-->>Agent: Generate response
    Agent->>User: Respond with enhanced information
```

### 2. Defining custom tools with typed parameters and validation

```mermaid
classDiagram
    class Tool {
        +name: str
        +description: str
        +parameters_model: Type[BaseModel]
        +as_llm_tool()
        +__call__(context, parameters, conversation)
    }
    
    class BaseModel {
        +model_validate()
        +model_dump()
    }
    
    class ParametersModel {
        +field1: Type1
        +field2: Type2
        +validate()
    }
    
    class ReturnModel {
        +result: Any
        +validate()
    }
    
    Tool --> ParametersModel : validates input
    Tool --> ReturnModel : returns
    ParametersModel --|> BaseModel : extends
    ReturnModel --|> BaseModel : extends
```

### 3. Managing conversation state and context across interactions

```mermaid
stateDiagram-v2
    [*] --> ConversationStart
    ConversationStart --> ReceiveUserMessage
    ReceiveUserMessage --> ProcessMessage
    ProcessMessage --> DecideTool
    
    DecideTool --> NoTool
    DecideTool --> UseTool
    
    UseTool --> ExecuteTool
    ExecuteTool --> IncorporateToolResult
    IncorporateToolResult --> GenerateResponse
    
    NoTool --> GenerateResponse
    GenerateResponse --> SendResponse
    SendResponse --> ReceiveUserMessage
    
    state "Conversation History" as CH
    ReceiveUserMessage --> CH : updates
    ProcessMessage --> CH : reads
    GenerateResponse --> CH : reads/updates
```

### 4. Structured communication between LLMs and external systems

```mermaid
flowchart TD
    User[User] --> Agent[Agent]
    Agent --> LLM[Language Model]
    LLM --> Response[Response Generator]
    
    Agent --> ToolRegistry[Tool Registry]
    ToolRegistry --> Tool1[Tool 1]
    ToolRegistry --> Tool2[Tool 2]
    ToolRegistry --> ToolN[Tool N]
    
    Tool1 --> ExternalSystem1[External System 1]
    Tool2 --> ExternalSystem2[External System 2]
    ToolN --> ExternalSystemN[External System N]
    
    ExternalSystem1 --> Tool1
    ExternalSystem2 --> Tool2
    ExternalSystemN --> ToolN
    
    Tool1 --> Response
    Tool2 --> Response
    ToolN --> Response
    
    Response --> Agent
    Agent --> User
```

### 5. Tool-assisted responses where the AI decides when to use tools

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM as "LLM Decision Making"
    participant Tools

    User->>Agent: "What's the weather in Paris?"
    Agent->>LLM: Process query with available tools
    
    Note over LLM: Decides this query requires a tool
    
    LLM-->>Agent: Choose "get_weather" tool
    Agent->>Tools: Call get_weather(location="Paris")
    Tools-->>Agent: Return weather data
    Agent->>LLM: Generate response with weather data
    LLM-->>Agent: "The weather in Paris is 72°F and sunny."
    Agent->>User: "The weather in Paris is 72°F and sunny."
    
    User->>Agent: "Thanks! How does that compare to typical April weather?"
    Agent->>LLM: Process follow-up query
    
    Note over LLM: Decides this can be answered without a tool
    
    LLM-->>Agent: Direct answer (no tool needed)
    Agent->>User: "April in Paris typically ranges from 50-65°F, so today is warmer than usual."
```

## Installation

```bash
pip install frizz
```

## Requirements

- Python e 3.12

## Core Concepts

### Agent

The `Agent` class manages conversations with language models and facilitates the use of tools in response to user queries. It handles the workflow of receiving user input, generating AI responses, and executing tool calls when appropriate.

### Tools

Tools are functions that provide specific functionality to the agent. Each tool:
- Has a name and description
- Accepts typed parameters (using Pydantic models)
- Returns structured data (also using Pydantic models)
- Can access a shared context object

### Usage Example

```python
from pydantic import BaseModel
from aikernel import Conversation, LLMUserMessage, LLMSystemMessage
from frizz import Agent, tool

# Define parameter and return models
class WeatherParams(BaseModel):
    location: str

class WeatherResult(BaseModel):
    temperature: float
    description: str

# Define a context type
class MyContext:
    def __init__(self, api_key: str):
        self.api_key = api_key

# Create a tool
@tool(name="get_weather")
async def get_weather(*, context: MyContext, parameters: WeatherParams, conversation: Conversation) -> WeatherResult:
    """Get the current weather for a location."""
    # Implementation using context.api_key to call a weather API
    return WeatherResult(temperature=72.5, description="Sunny")

# Create an agent with the tool
agent = Agent(
    tools=[get_weather],
    context=MyContext(api_key="your-api-key"),
    system_message=LLMSystemMessage(content="You are a helpful assistant.")
)

# Process a user message
from aikernel import LLMRouter, LLMModelAlias
user_message = LLMUserMessage(content="What's the weather in San Francisco?")
result = await agent.step(
    user_message=user_message,
    model="claude-3-sonnet-20240229",
    router=LLMRouter()
)
```

## License

[License information]