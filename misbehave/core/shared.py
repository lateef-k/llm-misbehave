"""
Shared message classes for the reveal framework.

This module contains all message-related classes and types used across
the LLM and Agent modules for consistent type handling.
"""

import typing as t
from dataclasses import dataclass

import logfire
import pydantic as pyd
from agents import TResponseInputItem
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.chat.chat_completion_message_param import \
    ChatCompletionMessageParam


class TFunctionCall(t.TypedDict):
    arguments: str
    name: str


@dataclass(frozen=True)
class Message:
    """A chat message with role and content."""

    role: str
    type: t.Literal[
        "text", "structured_output", "reasoning", "function_call", "function_output"
    ]
    name: t.Optional[str] = None

    content: t.Optional[str] = None
    reasoning: t.Optional[str] = None
    structured_output: t.Optional[dict] = None
    function_call: t.Optional[TFunctionCall] = None
    function_output: t.Optional[str] = None

    _raw: t.Optional[ChatCompletion | ParsedChatCompletion] = None
    _input: t.Optional[TResponseInputItem] = None

    def to_openai_param(self) -> ChatCompletionMessageParam:
        return t.cast(
            ChatCompletionMessageParam,
            dict(role=self.role, content=self.content, name=self.name),
        )

    @classmethod
    def user(cls, msg: str, name: t.Optional[str] = None):
        return TextMessage(role="user", content=msg, name=name)

    @classmethod
    def system(cls, msg: str):
        return TextMessage(role="system", content=msg)

    @classmethod
    def assistant(cls, msg: str):
        return TextMessage(role="assistant", content=msg)

    @classmethod
    def from_completion(
        cls, completion: ChatCompletion
    ) -> tuple[t.Optional["ReasoningMessage"], "TextMessage"]:
        message = completion.choices[0].message
        content = message.content or ""
        reasoning = message.reasoning if hasattr(message, "reasoning") else ""  # type: ignore

        reasoning_msg = None
        if reasoning:
            reasoning_msg = ReasoningMessage(
                role="assistant",
                reasoning=reasoning,
            )

        text_msg = TextMessage(role="assistant", content=content)
        return reasoning_msg, text_msg

    @classmethod
    def from_parsed_completion(
        cls,
        completion: ParsedChatCompletion,
        response_Format: t.Type[pyd.BaseModel],
    ) -> tuple[t.Optional["ReasoningMessage"], "StructuredOutputMessage"]:
        message = completion.choices[0].message
        reasoning = message.reasoning if hasattr(message, "reasoning") else ""  # type: ignore

        structured_output_dict = None
        if message.parsed:
            structured_output_dict = message.parsed.model_dump()
        elif message.tool_calls:
            structured_output_json = message.tool_calls[0].function.arguments
            structured_output_obj = response_Format.model_validate_json(
                structured_output_json
            )
            structured_output_dict = structured_output_obj.model_dump()
        else:
            structured_output_dict = None
            logfire.warn("No structured output for current completion.")

        reasoning_msg = None
        if reasoning:
            reasoning_msg = ReasoningMessage(
                role="assistant",
                reasoning=reasoning,
            )
        assert structured_output_dict
        structured_output_msg = StructuredOutputMessage(
            role="assistant", structured_output=structured_output_dict
        )
        return reasoning_msg, structured_output_msg


@dataclass(frozen=True)
class TextMessage(Message):
    content: str = ""
    type: t.Literal["text"] = "text"


@dataclass(frozen=True)
class ReasoningMessage(Message):
    reasoning: str = ""
    type: t.Literal["reasoning"] = "reasoning"


@dataclass(frozen=True)
class StructuredOutputMessage(Message):
    structured_output: t.Optional[dict] = None
    type: t.Literal["structured_output"] = "structured_output"


@dataclass(frozen=True)
class FunctionCallMessage(Message):
    function_call: t.Optional[TFunctionCall] = None
    type: t.Literal["function_call"] = "function_call"


@dataclass(frozen=True)
class FunctionOutputMessage(Message):
    function_output: t.Optional[str] = None
    type: t.Literal["function_output"] = "function_output"
