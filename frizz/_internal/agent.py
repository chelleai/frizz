import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property

import yaml
from aikernel import (
    Conversation,
    LLMAssistantMessage,
    LLMMessagePart,
    LLMModel,
    LLMSystemMessage,
    LLMToolMessage,
    LLMToolMessageFunctionCall,
    LLMUserMessage,
    llm_structured,
    llm_tool_call,
)
from pydantic import BaseModel, ValidationError

from frizz._internal.tools import Tool
from frizz._internal.types.response import AgentMessage
from frizz.errors import FrizzError


class Agent[ContextT]:
    def __init__(
        self,
        *,
        tools: list[Tool[ContextT, BaseModel, BaseModel]],
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

    @contextmanager
    def tool_aware_conversation(self) -> Iterator[None]:
        with self._conversation.with_temporary_system_message(message_part=self.__get_tools_system_message_part()):
            yield

    async def step(self, *, user_message: LLMUserMessage, model: LLMModel) -> None:
        with self.conversation.session():
            self._conversation.add_user_message(message=user_message)

            with self.tool_aware_conversation():
                agent_message = await llm_structured(
                    messages=self._conversation.render(),
                    model=model,
                    response_model=AgentMessage,
                )

            self._conversation.add_assistant_message(
                message=LLMAssistantMessage(parts=[LLMMessagePart(content=agent_message.structured_response.text)])
            )

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
                    )
                    parameters = chosen_tool.parameters_model.model_validate(parameters_response.tool_call.arguments)
                except ValidationError as error:
                    raise FrizzError(
                        f"Invalid tool parameters for tool {agent_message.structured_response.chosen_tool_name}: {error}"
                    )

                try:
                    result = await chosen_tool(context=self._context, parameters=parameters)
                except Exception as error:
                    raise FrizzError(
                        f"Error calling tool {agent_message.structured_response.chosen_tool_name}: {error}"
                    )

                self._conversation.add_tool_message(
                    tool_message=LLMToolMessage(
                        tool_call_id=parameters_response.tool_call.id,
                        name=parameters_response.tool_call.tool_name,
                        response=result.model_dump_json(),
                        function_call=LLMToolMessageFunctionCall(
                            name=parameters_response.tool_call.tool_name,
                            arguments=json.dumps(parameters_response.tool_call.arguments),
                        ),
                    )
                )

    def __get_tools_system_message_part(self) -> LLMMessagePart:
        return LLMMessagePart(
            content=f"""
            You have access to the following tools:
            {"\n".join([yaml.safe_dump(tool.as_llm_tool().render()) for tool in self._tools])}
        """
        )
