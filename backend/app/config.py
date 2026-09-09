"""Configuration. Every secret comes from the environment.

No key, password or connection string is ever written into the repository, and
``database_url_safe`` exists so connection details can be logged without
leaking credentials into a log aggregator.
"""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Required secrets -------------------------------------------------- #
    odds_api_key: str = Field(..., alias="ODDS_API_KEY")
    database_url: str = Field(..., alias="DATABASE_URL")

    # --- Optional providers ------------------------------------------------ #
    weather_api_key: str | None = Field(None, alias="WEATHER_API_KEY")
    espn_api_base: str = Field(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
        alias="ESPN_API_BASE",
    )
    pff_api_key: str | None = Field(None, alias="PFF_API_KEY")

    # --- Infrastructure ---------------------------------------------------- #
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"], alias="CORS_ORIGINS"
    )
    environment: str = Field("development", alias="ENVIRONMENT")

    # --- Model defaults ---------------------------------------------------- #
    odds_regions: str = Field("us,us2", alias="ODDS_REGIONS")
    default_simulations: int = Field(10_000, alias="DEFAULT_SIMULATIONS")
    devig_method: str = Field("multiplicative", alias="DEVIG_METHOD")
    kelly_fraction: float = Field(0.25, alias="KELLY_FRACTION")
    kelly_cap: float = Field(0.02, alias="KELLY_CAP")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("kelly_fraction")
    @classmethod
    def _reject_full_kelly(cls, value: float) -> float:
        if value > 0.5:
            raise ValueError(
                "KELLY_FRACTION above 0.5 is rejected. Full and near-full "
                "Kelly assume the model's probabilities are exactly right; a "
                "prop model's estimation error makes that assumption unsafe."
            )
        return value

    @property
    def database_url_safe(self) -> str:
        """Connection string with the password stripped, safe to log."""
        parts = urlsplit(self.database_url)
        if parts.password:
            netloc = f"{parts.username}:***@{parts.hostname}"
            if parts.port:
                netloc += f":{parts.port}"
            parts = parts._replace(netloc=netloc)
        return urlunsplit(parts)


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover
        missing = [k for k in ("ODDS_API_KEY", "DATABASE_URL")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            ) from exc
        raise
