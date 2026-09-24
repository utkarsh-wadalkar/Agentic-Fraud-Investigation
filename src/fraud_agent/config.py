"""Runtime configuration sourced from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    source_dir: Path = Field(default_factory=lambda: Path("Drive Files"))
    artifact_dir: Path = Field(default_factory=lambda: Path(".artifacts"))
    cases_dir: Path = Field(default_factory=lambda: Path("cases"))
    tg_host: str = ""
    tg_graphname: str = "FraudGraph"
    tg_secret: SecretStr = SecretStr("")
    tg_api_token: SecretStr = SecretStr("")
    tg_tgcloud: bool = True
    anthropic_base_url: str = ""
    anthropic_auth_token: SecretStr = SecretStr("")
    anthropic_model: str = ""

    @property
    def tigergraph_ready(self) -> bool:
        credential = self.tg_secret.get_secret_value() or self.tg_api_token.get_secret_value()
        return bool(self.tg_host and self.tg_graphname and credential)

    @property
    def anthropic_ready(self) -> bool:
        return bool(
            self.anthropic_base_url
            and self.anthropic_auth_token.get_secret_value()
            and self.anthropic_model
        )
