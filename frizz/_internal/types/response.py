from aikernel import LLMAssistantMessage, LLMToolMessage
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Respond to the user's message as an agent.

    You also have a set of tools available to you. In `chosen_tool_name`, specify the name of
    a tool if you feel that your response should include the use of a tool.

    For example, if you want to ask the user a skill testing question, and you have a tool available
    for generating quiz questions, you might respond with the text "Let me ask you a question", and
    then choose the quiz question generating tool.
    """

    text: str = Field(description="The text of your response to the user's message.")
    chosen_tool_name: str | None = Field(description="The exact name of the tool you want to use, if any.")


class StepResult(BaseModel):
    assistant_message: LLMAssistantMessage = Field(description="The assistant message to add to the conversation.")
    tool_message: LLMToolMessage | None = Field(description="The tool messages to add to the conversation.")
