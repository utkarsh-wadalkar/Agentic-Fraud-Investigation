from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

from tests.test_domain import make_answer


def module():
    return importlib.import_module("fraud_agent.narrative")


class FakeMessages:
    async def create(self, **kwargs):
        payload = '{"summary":"Grounded summary.","sar_narrative":null}'
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=payload)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def test_anthropic_narrator_parses_guarded_prose_and_usage() -> None:
    client = SimpleNamespace(messages=FakeMessages())
    narrator = module().AnthropicNarrator(client=client, model="combo")

    summary, sar, tokens = asyncio.run(narrator.rewrite(make_answer()))

    assert summary == "Grounded summary."
    assert sar is None
    assert tokens == 120


class FailingMessages:
    def __init__(self) -> None:
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        raise RuntimeError("provider unavailable")


def test_anthropic_narrator_falls_back_when_provider_call_fails() -> None:
    messages = FailingMessages()
    answer = make_answer()
    narrator = module().AnthropicNarrator(
        client=SimpleNamespace(messages=messages), model="combo"
    )

    summary, sar, tokens = asyncio.run(narrator.rewrite(answer))

    assert summary == answer.case.summary
    assert sar is None
    assert tokens == 0
    assert messages.attempts == 2
