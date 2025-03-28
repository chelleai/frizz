from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import Any

import yaml
from aikernel import (
    Conversation,
    LLMAssistantMessage,
    LLMMessagePart,
    LLMModelAlias,
    LLMSystemMessage,
    LLMToolMessage,
    LLMToolMessageFunctionCall,
    LLMUserMessage,
    LLMRouter,
    llm_structured,
    llm_tool_call,
)
from pydantic import BaseModel, ValidationError

from frizz._internal.tools import Tool
from frizz._internal.types.response import AgentMessage, StepResult
from frizz._internal.types.system import IGetToolSystemMessagePart
from frizz.errors import FrizzError


class Agent[ContextT]:
    def __init__(
        self,
        *,
        tools: list[Tool[ContextT, Any, Any]],
        context: ContextT,
        system_message: LLMSystemMessage | None = None,
        conversation_dump: str | None = None,
        get_tools_system_message_part: IGetToolSystemMessagePart | None = None,
    ) -> None:
        self._tools = tools
        self._context = context
        self._conversation = (
            Conversation.load(dump=conversation_dump) if conversation_dump is not None else Conversation()
        )
        if system_message is not None:
            self._conversation.set_system_message(message=system_message)

        self._get_tools_system_message_part = get_tools_system_message_part or _default_get_tools_system_message_part

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @cached_property
    def tools_by_name(self) -> dict[str, Tool[ContextT, BaseModel, BaseModel]]:
        return {tool.name: tool for tool in self._tools}

    @contextmanager
    def tool_aware_conversation(self) -> Iterator[None]:
        message_part = self._get_tools_system_message_part(tools=self._tools)
        with self._conversation.with_temporary_system_message(message_part=message_part):
            yield

    async def step[M: LLMModelAlias](self, *, user_message: LLMUserMessage, model: M, router: LLMRouter[M]) -> StepResult:
        with self.conversation.session():
            self._conversation.add_user_message(message=user_message)

            with self.tool_aware_conversation():
                agent_message = await llm_structured(
                    messages=self._conversation.render(),
                    model=model,
                    response_model=AgentMessage,
                    router=router,
                )

            assistant_message = LLMAssistantMessage(
                parts=[LLMMessagePart(content=agent_message.structured_response.text)]
            )
            self._conversation.add_assistant_message(message=assistant_message)

            if agent_message.structured_response.chosen_tool_name is not None:
                chosen_tool = self.tools_by_name.get(agent_message.structured_response.chosen_tool_name)
                if chosen_tool is None:
                    raise FrizzError(f"Tool {agent_message.structured_response.chosen_tool_name} not found")

                try:
                    parameters_response = await llm_tool_call(
                        messages=self._conversation.render(),
                        model=model,
                        tools=[chosen_tool.as_llm_tool()],
                        tool_choice="required",
                        router=router,
                    )
                    parameters = chosen_tool.parameters_model.model_validate(parameters_response.tool_call.arguments)
                except ValidationError as error:
                    raise FrizzError(
                        f"Invalid tool parameters for tool {agent_message.structured_response.chosen_tool_name}: {error}"
                    )

                try:
                    result = await chosen_tool(
                        context=self._context, parameters=parameters, conversation=self._conversation
                    )
                except Exception as error:
                    raise FrizzError(
                        f"Error calling tool {agent_message.structured_response.chosen_tool_name}: {error}"
                    )

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
            else:
                tool_message = None

        return StepResult(assistant_message=assistant_message, tool_message=tool_message)


def _default_get_tools_system_message_part(*, tools: list[Tool[Any, Any, Any]]) -> LLMMessagePart:
    return LLMMessagePart(
        content=f"""
        Tools are available for use in this conversation.

        When using a tool, your message to the user should indicate to them that you are going to use that tool.
        Don't use the term "tool", since they don't know what that is. For example, if you have a tool to
        get the weather, you might say "let me check the weather".

        When calling a tool, do not include the tool name or any part of the call in the message to the user.
        Only include the tool name in the chosen tool field of the AgentMessage.

        You have access to the following tools:
        {"\n".join([yaml.safe_dump(tool.as_llm_tool().render()) for tool in tools])}
    """
    )
