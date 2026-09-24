"""Optional LLM prose synthesis constrained to validated investigation facts."""

# ruff: noqa: E501 -- prompt prose remains auditable as complete sentences.

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

from fraud_agent.config import Settings
from fraud_agent.models import InvestigationAnswer


class AnthropicNarrator:
    def __init__(self, client: Any, model: str, max_tokens: int = 1200) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> AnthropicNarrator:
        if not settings.anthropic_ready:
            raise ValueError("Anthropic endpoint is not configured")
        client = AsyncAnthropic(
            auth_token=settings.anthropic_auth_token.get_secret_value(),
            base_url=settings.anthropic_base_url,
        )
        return cls(client=client, model=settings.anthropic_model)

    async def rewrite(self, answer: InvestigationAnswer) -> tuple[str, str | None, int]:
        prompt = (
            "Rewrite only the analyst summary and, when present, the SAR narrative. "
            "Use only facts and IDs in the validated JSON. Do not change decisions, probabilities, "
            "actions, amounts, dates, or evidence. Return strict JSON with keys summary and "
            "sar_narrative. The summary must be 2-6 sentences. A filed SAR must be 6-12 sentences "
            "covering who, what, when, where, how, and why; otherwise sar_narrative must be null.\n\n"
            + answer.model_dump_json(indent=2)
        )
        tokens = 0
        for attempt in range(2):
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system="You are a grounded bank fraud-investigation writer. Output JSON only.",
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as error:
                prompt = (
                    f"The provider call failed ({error}). Retry once and return only strict JSON "
                    "with summary and sar_narrative, preserving every supplied fact."
                )
                if attempt == 1:
                    break
                continue
            usage = getattr(message, "usage", None)
            tokens += int(getattr(usage, "input_tokens", 0)) + int(
                getattr(usage, "output_tokens", 0)
            )
            text = "".join(
                str(getattr(block, "text", ""))
                for block in getattr(message, "content", [])
                if getattr(block, "type", "text") == "text"
            ).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                payload = json.loads(text)
                summary = str(payload["summary"]).strip()
                sar = payload.get("sar_narrative")
                if not summary:
                    raise ValueError("summary is empty")
                if answer.sar.file and not sar:
                    raise ValueError("filed SAR narrative is empty")
                return summary, str(sar).strip() if sar else None, tokens
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                prompt = (
                    f"Your previous output was invalid ({error}). Return only strict JSON with "
                    "summary and sar_narrative, preserving every supplied fact."
                )
                if attempt == 1:
                    break
        return answer.case.summary, answer.sar.narrative if answer.sar.file else None, tokens
