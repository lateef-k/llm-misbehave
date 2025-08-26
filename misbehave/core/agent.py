import asyncio
import typing as t

import logfire
from agents import (Agent, Model, ModelProvider, OpenAIChatCompletionsModel,
                    RunConfig, Runner, StopAtTools, Tool, TResponseInputItem,
                    function_tool, set_default_openai_api,
                    set_default_openai_client, set_tracing_disabled)
from openai import AsyncOpenAI
from openai.types.responses import response_input_param

from reveal.core import shared
from reveal.core.shared import (FunctionCallMessage, FunctionOutputMessage,
                                Message, ReasoningMessage, TextMessage)
from reveal.settings import Settings

client = AsyncOpenAI(
    base_url=Settings.openrouter_base_url,
    api_key=Settings.openrouter_api_key,
    timeout=200000,
)

set_default_openai_client(client)
set_default_openai_api("chat_completions")

set_tracing_disabled(disabled=True)


class OpenRouterProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        model = OpenAIChatCompletionsModel(model="gpt-oss-20b", openai_client=client)
        old_fetch = model._fetch_response

        # patch the fetch so reasoning doesn't get discarded
        async def patched_fetch_response(*args, **kwargs):
            response = await old_fetch(*args, **kwargs)
            for choice in response.choices:
                choice.message.reasoning_content = choice.message.reasoning
            return response

        model._fetch_response = patched_fetch_response
        return model


type AgentOutput = Message


class AgentClient:

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        stop_at_tool_names: t.Optional[list[str]] = None,
    ):
        self.agent = Agent(
            name=name,
            instructions=system_prompt,
            tools=tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=stop_at_tool_names or []),
        )
        self.run_config = RunConfig(model_provider=OpenRouterProvider())

    async def run(
        self, message: str | list[TResponseInputItem]
    ) -> t.AsyncGenerator[shared.Message, None]:
        max_retries = 3
        retry_delay = 2

        accumulated = []

        for attempt in range(max_retries):
            try:
                result = await Runner.run(
                    self.agent,
                    message if not accumulated else accumulated,
                    run_config=self.run_config,
                    max_turns=200,
                )

                # Success - process results
                for item in result.new_items:
                    stream_event = item.to_input_item()

                    # Convert stream events to Message objects
                    converted_message = None
                    match stream_event:
                        case {"id": _, "type": "message", "content": content, **rest}:
                            # Handle content being a list or string
                            content_str = (
                                content if isinstance(content, str) else str(content)
                            )
                            converted_message = TextMessage(
                                role="assistant",
                                content=content_str,
                                _input=stream_event,
                            )
                        case {"type": "reasoning", "summary": reasoning, **rest}:
                            # Handle reasoning being a list or string
                            reasoning_str = (
                                reasoning
                                if isinstance(reasoning, str)
                                else str(reasoning)
                            )
                            converted_message = ReasoningMessage(
                                role="assistant",
                                reasoning=reasoning_str,
                                _input=stream_event,
                            )
                        case {
                            "type": "function_call",
                            "name": name,
                            "arguments": arguments,
                            **rest,
                        }:
                            converted_message = FunctionCallMessage(
                                role="assistant",
                                function_call={"name": name, "arguments": arguments},
                                _input=stream_event,
                            )
                        case {"type": "function_call_output", "output": output, **rest}:
                            converted_message = FunctionOutputMessage(
                                role="assistant",
                                function_output=output,
                                _input=stream_event,
                            )
                        case _:
                            # Yield the original stream event for any unhandled types
                            breakpoint()
                            continue

                    assert converted_message
                    accumulated.append(converted_message._input)
                    yield converted_message
                break  # Exit retry loop on success

            except Exception as e:
                if "Tool" in str(e) and "not found" in str(e):
                    # Log the specific tool name error
                    logfire.error(f"Tool name error attempt {attempt + 1}: {e}")
                    accumulated.append(
                        response_input_param.Message(
                            content=[
                                {
                                    "text": f"Your tool use is incorrect and has produced the following error: {e}",
                                    "type": "input_text",
                                }
                            ],
                            role="developer",
                            status="completed",
                            type="message",
                        )
                    )

                    if attempt < max_retries - 1:
                        await asyncio.sleep(
                            retry_delay * (2**attempt)
                        )  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed, re-raise
                        raise
                else:
                    # Non-tool errors, re-raise immediately
                    raise


async def main():

    @function_tool
    def get_weather(city: str):
        print(f"[debug] getting weather for {city}")
        return f"The weather in {city} is sunny."

    # This will use the custom model provider
    agent_client = AgentClient(
        name="Assistant",
        system_prompt="You only respond in haikus. Always reason before answering",
        tools=[get_weather],
    )
    async for message in agent_client.run("What's the weather in Tokyo?"):
        print(message)
    breakpoint()


if __name__ == "__main__":
    asyncio.run(main())
