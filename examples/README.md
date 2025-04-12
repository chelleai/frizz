# Frizz Examples

This directory contains example code demonstrating the key capabilities of the Frizz package.

> **Important Note About API Compatibility and Configuration**  
> The examples in this directory demonstrate the conceptual capabilities of Frizz, but there are two issues that prevent them from running directly:
> 
> 1. **API Mismatch**: There's an API version mismatch between the Frizz package and its aikernel dependency:
>    - Messages in aikernel now require using the `parts` parameter instead of `content`
>    - Router creation requires using the `get_router` function with specific models
>    - The `Agent.step()` method passes a `model` parameter to `llm_structured()`, but the current version of that function doesn't accept this parameter
>
> 2. **Model Access Configuration**: The aikernel package is configured to use cloud-based LLM services (AWS Bedrock, etc.) which require:
>    - Proper credentials and configuration for the cloud services
>    - Access to specific model profiles and throughput configurations
>
> To run these examples, you would need to:
> 1. Update the API usage to match the current implementation
> 2. Configure proper credentials for accessing the LLM services
> 
> A patched example demonstrating the API fixes can be found in [patched_agent_example.py](patched_agent_example.py), but you would still need to configure proper cloud credentials to run it successfully.

## Examples Overview

1. **Basic Assistant** - Creating AI assistants that can call functions during conversations
   - Simple calculator assistant example
   - Shows basic tool integration with an agent

2. **Custom Tools** - Defining custom tools with typed parameters and validation
   - Music recommendation assistant with complex parameter validation
   - Demonstrates Pydantic model validation and enum usage

3. **Conversation State** - Managing conversation state and context across interactions
   - Shopping cart example that persists state between messages
   - Shows how context objects store information during conversations

4. **External Systems** - Structured communication between LLMs and external systems
   - Weather service integration example (with mock implementation)
   - Shows how to connect agents to external APIs and services

5. **AI Tool Decision** - Tool-assisted responses where the AI decides when to use tools
   - Multi-tool assistant that chooses between fact lookup, calculator, and joke tools
   - Demonstrates how the AI can intelligently decide which tool to use (or none)

## Running the Examples

Each example is self-contained and can be run directly:

```bash
# Make sure you have the frizz package installed
pip install frizz

# Run any example
python examples/1_basic_assistant/basic_assistant.py
```

You'll need to provide your own LLM API credentials depending on your setup.

## Learning Path

For best results, go through the examples in order, as they build on concepts from previous examples:

1. First, understand how to create a basic agent with a simple tool
2. Then, explore more complex tools with proper type validation
3. Learn how to maintain state across a multi-turn conversation
4. See how to integrate external services and APIs
5. Finally, see how an agent can intelligently choose between multiple tools