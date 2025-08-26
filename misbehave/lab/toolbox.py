"""Toolbox for LLM experiment mutations and persona interactions"""

import dataclasses as dc
import random
import re
import typing as t
from itertools import product

from agents import function_tool

from reveal.core.llm import LLMClient
from reveal.core.shared import Message


@dc.dataclass(frozen=True)
class FixedMutationPoint:
    name: str
    values: list[str]


@dc.dataclass(frozen=True)
class LLMMutationPoint:

    name: str
    prompt: str


@dc.dataclass
class PromptTemplate:

    template: str
    mutations: dict[str, FixedMutationPoint | LLMMutationPoint]
    llm_client: t.Optional[LLMClient] = None

    async def compute_variations(self) -> list["MutatedPrompt"]:
        mutation_names = list(self.mutations.keys())
        mutation_values = []

        for name in mutation_names:
            mutation = self.mutations[name]
            if isinstance(mutation, FixedMutationPoint):
                mutation_values.append(mutation.values)
            elif isinstance(mutation, LLMMutationPoint):
                if self.llm_client is None:
                    raise ValueError(
                        f"LLM client required for LLM mutation point '{name}'"
                    )
                reasoning_msg, text_msg = await self.llm_client.complete(
                    [Message.system(mutation.prompt)]
                )
                assert text_msg and (variation := text_msg.content)
                mutation_values.append(
                    [v.strip() for v in variation.split("\n") if v.strip()]
                )
            else:
                raise ValueError(f"Unknown mutation point type: {type(mutation)}")

        results = []
        for combination in product(*mutation_values):
            applied_mutations = dict(zip(mutation_names, combination))
            result = self.template

            for name, value in applied_mutations.items():
                result = result.replace("{" + name + "}", value)
                result = result.replace("{{" + name + "}}", value)

            results.append(
                MutatedPrompt(
                    prompt=result,
                    mutations_applied=applied_mutations,
                    template_id=id(self),
                )
            )

        return results


@dc.dataclass
class MutatedPrompt:

    prompt: str
    mutations_applied: dict[str, str]
    template_id: int

    @property
    def mutation_id(self) -> str:
        """Unique ID based on applied mutations"""
        sorted_mutations = sorted(self.mutations_applied.items())
        return "_".join(f"{k}={v}" for k, v in sorted_mutations)


class PersonaAgent:

    def __init__(self, llm_client: LLMClient, persona_prompt: str):
        self.llm_client = llm_client
        self.conversation: list[Message] = [Message.system(persona_prompt)]
        self.initial_message: t.Optional[str] = None

    async def get_initial_message(self) -> str:
        """Get the persona's opening message"""
        if self.initial_message is None:
            reasoning_msg, text_msg = await self.llm_client.complete(self.conversation)
            assert text_msg and (initial_message := text_msg.content)
            self.initial_message = initial_message
            self.conversation.append(text_msg)
            if not self.initial_message:
                breakpoint()
        return self.initial_message

    async def respond(self, message: str) -> str:
        """Get the persona's response to a message"""
        self.conversation.append(Message.user(message))
        reasoning_msg, text_msg = await self.llm_client.complete(self.conversation)
        assert text_msg
        self.conversation.append(text_msg)
        return text_msg.content
