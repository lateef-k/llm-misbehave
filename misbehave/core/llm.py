"""
OpenAI-compatible LLM client for OpenRouter API.

This module provides a well-typed, dataclass-based interface for interacting
with OpenRouter's API using the OpenAI client library.
"""

import asyncio
import hashlib
import json
import logging
import typing as t

import pydantic as pyd
from openai import NOT_GIVEN, AsyncOpenAI, NotGiven
from rich import print

from reveal.core.cache import AsyncCache, init_db
from reveal.core.shared import (Message, ReasoningMessage,
                                StructuredOutputMessage, TextMessage)
from reveal.settings import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client for OpenRouter API."""

    def __init__(
        self,
        base_url: str = Settings.openrouter_base_url,
        api_key: str = Settings.openrouter_api_key,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0,
        max_tokens: int | NotGiven = NOT_GIVEN,
        timeout: float = 60.0,
        max_retries: int = 3,
        top_logprobs: t.Optional[int] = None,
        effort: t.Literal["minimal", "low", "medium", "high"] = "medium",
        extra_body: t.Optional[dict] = None,
        extra_headers: t.Optional[dict] = None,
        cache_enabled: bool = True,
    ):
        """Initialize the LLM client with configuration."""
        if not extra_body:
            extra_body = {}
        if not extra_headers:
            extra_headers = {}

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logprobs = True if top_logprobs else False
        self.top_logprobs = top_logprobs
        self.effort: t.Literal["minimal", "low", "medium", "high"] = effort
        self.extra_body = extra_body | {
            "effort": self.effort,
        }
        self.extra_headers = extra_headers | {
            # "X-OpenRouter-Provider-Exclude": "Fireworks"  # no structured output
        }
        self.cache_enabled = cache_enabled
        self._cache = AsyncCache()

    async def _ensure_cache_initialized(self) -> None:
        await init_db()

    def _generate_cache_key(
        self,
        messages: list[Message],
        response_format: t.Optional[t.Type[pyd.BaseModel]] = None,
    ) -> str:
        cache_data = {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if self.max_tokens != NOT_GIVEN else None,
            "effort": self.effort,
            "top_logprobs": self.top_logprobs,
            "response_format": response_format.__name__ if response_format else None,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    async def complete(
        self,
        messages: list[Message],
    ) -> tuple[t.Optional[ReasoningMessage], t.Optional[TextMessage]]:
        """Create an async chat completion."""
        await self._ensure_cache_initialized()

        if self.cache_enabled:
            cache_key = self._generate_cache_key(messages)
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[msg.to_openai_param() for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            logprobs=self.logprobs,
            reasoning_effort=self.effort,
            extra_body=self.extra_body,
        )

        reasoning_msg, text_msg = Message.from_completion(response)

        if self.cache_enabled:
            await self._cache.set(cache_key, (reasoning_msg, text_msg))  # type: ignore

        return reasoning_msg, text_msg

    async def parse[T: pyd.BaseModel](
        self, messages: list[Message], response_format: t.Type[T], recompute=False
    ) -> tuple[t.Optional[ReasoningMessage], t.Optional[StructuredOutputMessage]]:
        await self._ensure_cache_initialized()

        if self.cache_enabled and not recompute:
            cache_key = self._generate_cache_key(messages)
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        response = await self._async_client.chat.completions.parse(
            model=self.model,
            messages=[msg.to_openai_param() for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            logprobs=self.logprobs,
            reasoning_effort=self.effort,
            response_format=response_format,
            extra_body=self.extra_body,
            extra_headers=self.extra_headers,
        )

        reasoning_msg, structured_output_msg = Message.from_parsed_completion(
            response, response_format
        )

        if self.cache_enabled or recompute:
            cache_key = self._generate_cache_key(messages)
            await self._cache.set(cache_key, (reasoning_msg, structured_output_msg))  # type: ignore

        return reasoning_msg, structured_output_msg


async def main():
    class ExampleName(pyd.BaseModel):
        human_name: str

    client = LLMClient(effort="medium", cache_enabled=False)
    # result = await client.complete(
    #     [Message.user("Proof the collatz conjecture or at least try to")]
    # )
    result = await client.parse(
        [Message.user("Give me an example of a name")], response_format=ExampleName
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
