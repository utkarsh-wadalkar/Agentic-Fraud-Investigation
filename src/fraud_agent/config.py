"""Runtime configuration sourced from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    source_dir: Path = Field(default_factory=lambda: Path("Drive Files"))
    artifact_dir: Path = Field(default_factory=lambda: Path(".artifacts"))
    cases_dir: Path = Field(default_factory=lambda: Path("cases"))
    tg_host: str = Field(default_factory=lambda: os.getenv("TG_HOST", ""))
    tg_graphname: str = Field(default_factory=lambda: os.getenv("TG_GRAPHNAME", "FraudGraph"))
    tg_api_token: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("TG_API_TOKEN", ""))
    )
    tg_tgcloud: bool = Field(
        default_factory=lambda: os.getenv("TG_TGCLOUD", "true").lower() == "true"
    )
    anthropic_base_url: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "")
    )
    anthropic_auth_token: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_AUTH_TOKEN", ""))
    )
    anthropic_model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", ""))

    @property
    def tigergraph_ready(self) -> bool:
        return bool(self.tg_host and self.tg_graphname and self.tg_api_token.get_secret_value())

    @property
    def anthropic_ready(self) -> bool:
        return bool(
            self.anthropic_base_url
            and self.anthropic_auth_token.get_secret_value()
            and self.anthropic_model
        )
