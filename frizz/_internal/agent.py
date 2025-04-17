from functools import cached_property
from typing import Any

from pydantic import BaseModel, ValidationError

from aikernel import (
    Conversation,
    LLMAssistantMessage,
    LLMMessagePart,
    LLMModelName,
    LLMRouter,
    LLMSystemMessage,
    LLMToolMessage,
    LLMToolMessageFunctionCall,
    LLMUserMessage,
    llm_tool_call,
)
from frizz._internal.tools import Tool
from frizz._internal.types.response import StepResult
from frizz.errors import FrizzError


class Agent[ContextT]:
    def __init__(
        self,
        *,
        tools: list[Tool[ContextT, Any, Any]],
        context: ContextT,
        system_message: LLMSystemMessage | None = None,
        conversation_dump: str | None = None,
    ) -> None:
        self._tools = tools
        self._context = context
        self._conversation = (
            Conversation.load(dump=conversation_dump) if conversation_dump is not None else Conversation()
        )
        if system_message is not None:
            self._conversation.set_system_message(message=system_message)

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @cached_property
    def tools_by_name(self) -> dict[str, Tool[ContextT, BaseModel, BaseModel]]:
        return {tool.name: tool for tool in self._tools}

    async def step(self, *, user_message: LLMUserMessage, router: LLMRouter[LLMModelName]) -> StepResult:
        with self.conversation.session():
            self._conversation.add_user_message(message=user_message)

            agent_response = await llm_tool_call(
                messages=self._conversation.render(),
                router=router,
                tools=[tool.as_llm_tool() for tool in self._tools],
                tool_choice="auto",
            )

            if agent_response.text is not None:
                assistant_message = LLMAssistantMessage(parts=[LLMMessagePart(content=agent_response.text)])
                self._conversation.add_assistant_message(message=assistant_message)
                tool_message = None
            elif agent_response.tool_call is not None:
                chosen_tool = self.tools_by_name.get(agent_response.tool_call.tool_name)
                if chosen_tool is None:
                    raise FrizzError(f"Tool {agent_response.tool_call.tool_name} not found")

                parameters_response = await llm_tool_call(
                    messages=self._conversation.render(),
                    router=router,
                    tools=[chosen_tool.as_llm_tool()],
                    tool_choice="required",
                )

                try:
                    parameters = chosen_tool.parameters_model.model_validate(parameters_response.tool_call.arguments)
                except ValidationError as error:
                    raise FrizzError(f"Invalid tool parameters for tool {agent_response.tool_call.tool_name}: {error}")

                try:
                    result = await chosen_tool(
                        context=self._context,
                        parameters=parameters,
                        conversation=self._conversation,
                    )
                except Exception as error:
                    raise FrizzError(f"Error calling tool {agent_response.tool_call.tool_name}: {error}")

                tool_message = LLMToolMessage(
                    tool_call_id=parameters_response.tool_call.id,
                    name=parameters_response.tool_call.tool_name,
                    response=result.model_dump(),
                    function_call=LLMToolMessageFunctionCall(
                        name=parameters_response.tool_call.tool_name,
                        arguments=parameters_response.tool_call.arguments,
                    ),
                )
                self._conversation.add_tool_message(tool_message=tool_message)
                assistant_message = None
            else:
                raise RuntimeError("Not a possible state, validated by Pydantic")

        return StepResult(assistant_message=assistant_message, tool_message=tool_message)
