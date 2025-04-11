"""
Example 5: Tool-assisted responses where the AI decides when to use tools

This example demonstrates how an AI assistant can intelligently decide when to use tools
during a conversation, choosing between multiple available tools or direct responses.
"""
import asyncio
import re
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from aikernel import Conversation, LLMUserMessage, LLMSystemMessage, LLMRouter
from frizz import Agent, tool


# Models for different tools the AI can choose between
class FactLookupParams(BaseModel):
    """Parameters for fact lookup tool."""
    query: str = Field(..., description="The fact to look up")


class FactResult(BaseModel):
    """Result from fact lookup."""
    fact: str
    source: Optional[str] = None
    confidence: float


class CalculatorParams(BaseModel):
    """Parameters for calculator tool."""
    expression: str = Field(..., description="Mathematical expression to calculate")


class CalculatorResult(BaseModel):
    """Result from calculator."""
    expression: str
    result: float
    steps: Optional[List[str]] = None


class JokeParams(BaseModel):
    """Parameters for joke tool."""
    topic: str = Field(..., description="Topic for the joke")
    safe_mode: bool = Field(True, description="Whether to ensure the joke is family-friendly")


class JokeResult(BaseModel):
    """Result from joke generator."""
    joke: str
    type: str = "one-liner"  # could be "one-liner", "pun", "riddle", etc.


# Mock knowledge base for our fact lookup tool
FACT_DATABASE = {
    "earth": {
        "fact": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
        "source": "NASA",
        "confidence": 1.0
    },
    "mars": {
        "fact": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury.",
        "source": "NASA",
        "confidence": 1.0
    },
    "jupiter": {
        "fact": "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
        "source": "NASA",
        "confidence": 1.0
    },
    "water": {
        "fact": "Water covers about 71% of the Earth's surface.",
        "source": "USGS",
        "confidence": 0.99
    },
    "human body": {
        "fact": "The adult human body consists of approximately 60% water.",
        "source": "Medical literature",
        "confidence": 0.95
    },
    "eiffel tower": {
        "fact": "The Eiffel Tower was completed in 1889 and stands 324 meters (1,063 ft) tall.",
        "source": "Official Eiffel Tower website",
        "confidence": 1.0
    }
}


# Mock joke database
JOKE_DATABASE = {
    "programming": [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "A SQL query walks into a bar, approaches two tables and asks: 'Can I join you?'"
    ],
    "math": [
        "Why was six afraid of seven? Because seven eight nine!",
        "I'll do algebra, I'll do trigonometry, I'll even do statistics. But graphing is where I draw the line."
    ],
    "food": [
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "What do you call a fake noodle? An impasta!"
    ],
    "animals": [
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't scientists trust atoms? Because they make up everything!"
    ],
    "default": [
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "I'm on a seafood diet. Every time I see food, I eat it!"
    ]
}


# Context for our multi-tool example
class MultiToolContext:
    """Context with multiple tool implementations."""
    
    def lookup_fact(self, query: str) -> FactResult:
        """Look up a fact in our database."""
        # Simple implementation that looks for keyword matches
        query = query.lower()
        
        for key, data in FACT_DATABASE.items():
            if key in query:
                return FactResult(
                    fact=data["fact"],
                    source=data["source"],
                    confidence=data["confidence"]
                )
        
        # Default response if no match found
        return FactResult(
            fact="I don't have specific information about that.",
            source=None,
            confidence=0.1
        )
    
    def calculate(self, expression: str) -> CalculatorResult:
        """Calculate the result of a mathematical expression."""
        # Clean the expression and make it safe to evaluate
        # In a real implementation, you would use a proper math library
        expression = expression.replace("^", "**")
        
        # SECURITY WARNING: This is NOT how you should implement a calculator in production
        # This is just for demonstration purposes. In real code, NEVER use eval() on user input
        try:
            # Simple calculator that shows its work
            steps = []
            
            # Handle addition and subtraction with steps
            if "+" in expression or "-" in expression:
                parts = re.split(r'([+\-])', expression)
                if parts[0] == '':  # Handle leading minus sign
                    parts = parts[1:]
                
                if len(parts) > 1:
                    result = float(eval(parts[0]))
                    steps.append(f"Start with {parts[0]} = {result}")
                    
                    for i in range(1, len(parts), 2):
                        op = parts[i]
                        number = float(eval(parts[i+1]))
                        if op == "+":
                            steps.append(f"Add {number}")
                            result += number
                        else:
                            steps.append(f"Subtract {number}")
                            result -= number
                        steps.append(f"Current result: {result}")
                else:
                    result = float(eval(expression))
            else:
                result = float(eval(expression))
                steps = [f"Evaluate {expression} = {result}"]
            
            return CalculatorResult(
                expression=expression,
                result=result,
                steps=steps
            )
        except Exception as e:
            return CalculatorResult(
                expression=expression,
                result=0.0,
                steps=[f"Error: {str(e)}"]
            )
    
    def generate_joke(self, topic: str, safe_mode: bool = True) -> JokeResult:
        """Generate a joke about the given topic."""
        topic_lower = topic.lower()
        
        # Find matching jokes from our database
        for key, jokes in JOKE_DATABASE.items():
            if key in topic_lower:
                joke = jokes[0]  # Just take the first one for simplicity
                return JokeResult(joke=joke, type="one-liner")
        
        # If no specific topic match, return a default joke
        return JokeResult(joke=JOKE_DATABASE["default"][0], type="one-liner")


# Define tools for the AI to choose from
@tool(name="lookup_fact")
async def lookup_fact(
    *, context: MultiToolContext, parameters: FactLookupParams, conversation: Conversation
) -> FactResult:
    """Look up a fact about the given query.
    
    Use this tool when the user asks for factual information about topics like
    astronomy, geography, biology, history, or other general knowledge.
    """
    return context.lookup_fact(parameters.query)


@tool(name="calculate")
async def calculate(
    *, context: MultiToolContext, parameters: CalculatorParams, conversation: Conversation
) -> CalculatorResult:
    """Calculate the result of a mathematical expression.
    
    Use this tool when the user wants to perform calculations, solve math problems,
    or evaluate mathematical expressions.
    """
    return context.calculate(parameters.expression)


@tool(name="tell_joke")
async def tell_joke(
    *, context: MultiToolContext, parameters: JokeParams, conversation: Conversation
) -> JokeResult:
    """Generate a joke about the given topic.
    
    Use this tool when the user explicitly asks for a joke or humor about a specific topic.
    """
    return context.generate_joke(parameters.topic, parameters.safe_mode)


async def main():
    # Create context and agent with multiple tools
    context = MultiToolContext()
    
    agent = Agent(
        tools=[lookup_fact, calculate, tell_joke],
        context=context,
        system_message=LLMSystemMessage(content="""
            You are a helpful assistant that can answer questions, solve problems, and tell jokes.
            You have several tools available:
            
            1. For factual questions, use the lookup_fact tool
            2. For mathematical calculations, use the calculate tool
            3. For jokes and humor, use the tell_joke tool
            
            For other types of questions or conversations, answer directly without using tools.
            Choose the most appropriate approach for each user request.
        """)
    )
    
    # Create a router for the LLM API
    router = LLMRouter()
    
    # Example conversation
    print("Starting conversation with the multi-tool assistant...\n")
    
    # Example 1: Factual question that should use the lookup tool
    user_message = LLMUserMessage(content="Can you tell me about Mars?")
    print(f"User: {user_message.content}")
    
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Example 2: Math question that should use the calculator tool
    user_message = LLMUserMessage(content="What's 235 + 489?")
    print(f"\nUser: {user_message.content}")
    
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Example 3: Request for a joke
    user_message = LLMUserMessage(content="Tell me a programming joke")
    print(f"\nUser: {user_message.content}")
    
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Example 4: General question that shouldn't use tools
    user_message = LLMUserMessage(content="How are you today?")
    print(f"\nUser: {user_message.content}")
    
    result = await agent.step(
        user_message=user_message,
        model="claude-3-sonnet-20240229",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")


if __name__ == "__main__":
    asyncio.run(main())