from pydantic import BaseModel, Field

from aikernel import LLMAssistantMessage, LLMToolMessage


class StepResult(BaseModel):
    assistant_message: LLMAssistantMessage | None = Field(
        description="The assistant message to add to the conversation."
    )
    tool_message: LLMToolMessage | None = Field(description="The tool messages to add to the conversation.")
